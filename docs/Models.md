# Models

#### Contents

1. [Supported Models](#Supported-Models)
1. [Inheritance Relationship](#Inheritance-Relationship)

## Supported Models

| Class         | Description    |Supported Algorithms |
| :------------- | :----------:| :----------:    |
| [BaseModel](../basicsr/models/base_model.py) | Abstract base class, define common functions||
| [SRModel](../basicsr/models/sr_model.py) | Base image SR class | SRCNN, EDSR, SRResNet, RCAN, RRDBNet, etc |
| [SRGANModel](../basicsr/models/srgan_model.py) | SRGAN image SR class | SRGAN |
| [ESRGANModel](../basicsr/models/esrgan_model.py) | ESRGAN image SR class|ESRGAN|
| [StyleGAN2Model](../basicsr/models/stylegan2_model.py) | StyleGAN2 generation class |StyleGAN2|

## Inheritance Relationship

In order to reuse components among models, we use a lot of inheritance. The following is the inheritance relationship:

```txt
BaseModel
├── SRModel
│   ├── SRGANModel
│   │   └── ESRGANModel
└── StyleGAN2Model
```
