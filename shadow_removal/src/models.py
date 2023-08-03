import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0

        self.model_save = config.PATH


    def load(self, model_path):
        self.gen_weights_path = model_path

        if os.path.exists(self.gen_weights_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.gen_weights_path)
            else:
                data = torch.load(self.gen_weights_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']

    def save(self, extend=''):

        if len(self.config.GPU) > 1:
            generate_param = self.generator.module.state_dict()
            print('save...multiple GPU')
        else:
            generate_param = self.generator.state_dict()
            print('save...single GPU')

        torch.save({
            'iteration': self.iteration,
            'generator': generate_param
        }, os.path.join(self.model_save, '{}_{}_gen_{}.pth'.format(self.iteration, self.name, extend)))

        print('\nsaving %s...\n' % self.name)


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        generator = InpaintGenerator(config=config)

        l1_loss = nn.L1Loss()

        self.add_module('generator', generator)
        self.add_module('l1_loss', l1_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

    def calculate_loss(self, outputs, images, masks):

        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)

        # create logs
        logs = [
            ("l_l1", gen_l1_loss.item()),
        ]

        return gen_l1_loss, logs

    def process(self, images, masks, shadows):
        self.iteration += 1

        self.gen_optimizer.zero_grad()

        outputs_down = self(shadows, masks)

        down_gen_loss, down_logs = self.calculate_loss(outputs=outputs_down, images=images, masks=masks)

        gen_loss = down_gen_loss
        logs = down_logs

        return outputs_down, outputs_down, gen_loss, logs

    def forward(self, shadows, masks):
        shadows_masked = shadows * (1 - masks)

        inputs_up = torch.cat((shadows_masked, masks), dim=1)
        inputs_down = torch.cat((shadows, masks), dim=1)

        outputs_down = self.generator(inputs_down, inputs_up)
        return outputs_down

    def backward(self, gen_loss=None):
        gen_loss.backward()
        self.gen_optimizer.step()

