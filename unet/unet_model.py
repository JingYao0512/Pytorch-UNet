""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module): # Pytorch中定義Model的基本類別
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        ## 定義模型參數
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        ## 定義需要的layer
        # Encoder 
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))

        # Decoder
        factor = 2 if bilinear else 1 ## 放大倍率
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes)) # 用於生成輸出的分割預測結果

    def forward(self, x):
        ## 定義模型的架構 (層與層的連線)
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def use_checkpointing(self):
        # 用於啟用模型的checkpointing功能
        # checkpointing是一種優化技術
        # 可以在前向傳播過程中節省內存
        # 特別適用於深層網絡
        # 這裡將模型中的每個卷積層都應用了checkpointing。
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)