import torch, torchvision
from torch import nn
from PIL import Image

import sys
import time,datetime,pytz

def get_device(i=0):
    """Return gpu(i) if exists, otherwise return cpu().
    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

#----------- get the contents and styles -----------
def extract_features(X, net, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

def get_contents(content_img, image_shape, content_layers, style_layers, net, device):
    content_X = preProcess(content_img, image_shape).to(device)
    content_Y, _ = extract_features(content_X, net, content_layers, style_layers)
    return content_X, content_Y

def get_styles(style_img, image_shape, content_layers, style_layers, net, device):
    style_X = preProcess(style_img, image_shape).to(device)
    _, style_Y = extract_features(style_X, net, content_layers, style_layers)
    return style_X, style_Y


#---------- define the loss function ---------
def content_loss(Y_hat, Y):
    return torch.square(Y_hat - Y.detach()).mean()

def gram(X):
    num_channels, n = X.shape[1], X.numel()//X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T)/(num_channels * n)

def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:,:,1:,:] - Y_hat[:, :, :-1,:]).mean() +
                  torch.abs(Y_hat[:,:,:,1:] - Y_hat[:, :, : , :-1]).mean())

content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


#--------------------------------------------------


class SynthesizedImage(nn.Module):
    def __init__(self, image_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*image_shape))

    def forward(self):
        return self.weight

def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer

def train(X, net, contents_Y, styles_Y, contents_layer, styles_layer, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, net, contents_layer, styles_layer)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 ==0:
            print(f"contents_l {sum(contents_l):.3f}, styles_l {sum(styles_l):.3f}, tv_l {tv_l:.3f}")
            postProcess(X)
    return X
        
                        

#----------------------------------------------
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preProcess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)
                                                

def postProcess(img):
    shanghai = pytz.timezone("Asia/Shanghai")
    now = datetime.datetime.now(shanghai)
    fmt = '%Y%m%d_%H%M%S'
    s = (now.strftime(fmt))[2:]
    file_name = 'output/' + s + '.jpeg'
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    pic = torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
    pic.save(file_name)




def main(argv):
    device = get_device()
    image_shape = (300,450)


    pretrained_net = torchvision.models.vgg19(pretrained=True)
    style_layers, content_layers = [0, 5, 10, 19, 28], [25]
    net = nn.Sequential(*[pretrained_net.features[i] for i in
                          range(max(content_layers + style_layers) + 1)])

    content_img = Image.open("contents/draft_paper.jpg")
    style_img = Image.open("styles/shuilian.jpeg")

    content_X, content_Y = get_contents(content_img, image_shape, content_layers, style_layers, net, device)
    _, styles_Y = get_styles(style_img, image_shape, content_layers, style_layers, net, device)
    output = train(content_X, net,  content_Y, styles_Y, content_layers, style_layers, device, 0.3, 500, 50)




if __name__ == "__main__":
    print("call main")
    main(sys.argv[1:])
    print("main finished")

