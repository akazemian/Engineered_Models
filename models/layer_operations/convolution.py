from models.layer_operations.preset_filters import filters
from torch import nn
from torch.nn import functional as F
import math



class Convolution(nn.Module):
    
    """
    Attributes
    ----------
    filter_type  
        The type of filter used for convolution. One of : random, curvature, 1x1
    
    curv_params
        the parametrs used to create the filters, applicable for curvature filters
        
    filter_size 
        The kernel size used in layer. 
    
    out_channels
        the number of filters used for convolution 
    """
    
    
    def __init__(self, filter_type:str,
                 curv_params:dict=None,
                 filter_size:int=None,
                 out_channels:int=None):
                
        super().__init__()
        
        self.out_channels = out_channels
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.curv_params = curv_params
    

    
    
    def extra_repr(self) -> str:
        return 'out_channels={out_channels}, kernel_size={filter_size}, filter_type:{filter_type},curv_params:{curv_params}'.format(**self.__dict__)
    
    
    
    def forward(self,x):
            
        
        in_channels = x.shape[1]

        
        if in_channels == 3: # for RGB input (the preset L1 filters are repeated across the 3 channels)
            w = filters(filter_type=self.filter_type,out_channels=self.out_channels,in_channels=1,
                         kernel_size=self.filter_size,curv_params=self.curv_params)
            weight = w.repeat(1,3,1,1)

            
        else: # for grayscale input
            weight = filters(filter_type=self.filter_type,out_channels=self.out_channels,in_channels=in_channels,
                         kernel_size=self.filter_size,curv_params=self.curv_params)
        
        weight = weight.cuda()
        x =  x.cuda()
        x = F.conv2d(x,weight=weight,padding=math.floor(weight.shape[-1] / 2))

        return x
    




        
        
        
        