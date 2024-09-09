import torch
import torch.nn as nn
import os

from torch.utils.data import DataLoader
from dataset import ECGDataset
from utils import split_data
import warnings
from tqdm import tqdm
from utils import cal_f1s, cal_aucs 
import numpy as np
import math
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from Embed import DataEmbedding, TokenEmbedding, PositionalEmbedding, XPOS
import time





class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer with dropout
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)        
        
        # Forward propagate GRU
        

        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet1d(nn.Module):
    def __init__(self, block, layers, input_channels=12, inplanes=64, num_classes=9):
        super(ResNet1d, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
class Residual_Conv_Mamba(nn.Module):
    def __init__(self,d_model, d_conv, d_ff, expand, e_layers=2,input_channels=12, inplanes=64, num_classes=9 ):
        super(Residual_Conv_Mamba, self).__init__()
        self.inplanes=inplanes
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            self._make_layer(BasicBlock1d, 128, 2, stride=2)
            )
        self.embedding = nn.Linear(1875, d_model)        
        # self.value_embedding = TokenEmbedding(c_in=128, d_model=d_model)
        # self.position_embedding = PositionalEncoding(model_dim=d_model, dropout=0.1)
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)
        self.Mambalayers = nn.ModuleList([ResidualBlock( d_model,  d_conv, d_ff, self.d_inner, self.dt_rank) for _ in range(e_layers)])
        self.norm = RMSNorm(d_model)

        self.fc = nn.Linear(d_model, num_classes, bias=False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)        
    def forward(self,x):
        x=self.resnet_half(x)    
        x=self.embedding(x)
        for layer in self.Mambalayers:
            x = layer(x)
        x = self.norm(x)
        x = self.fc(x.mean(dim=1))

        return x
class Residual_Conv_GRU_test(nn.Module):
    def __init__(self,input_size, batch_size,input_channels=12, inplanes=64, num_classes=9, GRU_hidden_size=128, GRU_num_layers=2):
        super().__init__()
        self.inplanes=inplanes
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            self._make_layer(BasicBlock1d, 128, 2, stride=2)
            )
        self.gru = nn.GRU(input_size = 128,
                        hidden_size = GRU_hidden_size, num_layers=GRU_num_layers, 
                        batch_first=True, dropout=0.1)
        self.attention = nn.MultiheadAttention(128,8)
        self.fc = nn.Linear(GRU_hidden_size, num_classes)
        self.num_layers = GRU_num_layers
        self.hidden_size = GRU_hidden_size
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def get_output_shape(self ,input_size,batch_size,input_channels):
        dim=batch_size,input_channels,input_size
        return self.resnet_half(torch.zeros(dim)).size(2)
    
    def forward(self,x, quantize=None):
        x=self.resnet_half(x)
        if torch.isnan(x).any():
            print(f"NaN detected after layer: ")
            

        x = x.permute(0, 2, 1)
        if quantize:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,dtype=torch.float16).to(x.device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)       
        # Forward propagate GRU
        
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        score=self.attention(out)
        out=out * score
        out = torch.sum(out, dim=1)
        # Decode the hidden state of the last time step
        
        out = self.fc(out)
        
        
        return out



class Residual_Conv_GRU(nn.Module):
    def __init__(self,input_size, batch_size,input_channels=12, inplanes=64, num_classes=9, GRU_hidden_size=128, GRU_num_layers=2):
        super().__init__()
        self.inplanes=inplanes
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            self._make_layer(BasicBlock1d, 128, 2, stride=2)
            )
        self.gru = nn.GRU(input_size = 128,
                        hidden_size = GRU_hidden_size, num_layers=GRU_num_layers, 
                        batch_first=True, dropout=0.1)
        self.fc = nn.Linear(GRU_hidden_size, num_classes)
        self.num_layers = GRU_num_layers
        self.hidden_size = GRU_hidden_size
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def get_output_shape(self ,input_size,batch_size,input_channels):
        dim=batch_size,input_channels,input_size
        return self.resnet_half(torch.zeros(dim)).size(2)
    
    def forward(self,x, quantize=None):
        x=self.resnet_half(x)
        if torch.isnan(x).any():
            print(f"NaN detected after layer: ")
            

        x = x.permute(0, 2, 1)
        if quantize:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size,dtype=torch.float16).to(x.device)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)       
        # Forward propagate GRU
        
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out
    

class Residual_Conv_LSTM(nn.Module):
    def __init__(self,input_size, batch_size,input_channels=12, inplanes=64, num_classes=9, hidden_size=128, num_layers=2):
        super().__init__()
        self.inplanes=inplanes
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            self._make_layer(BasicBlock1d, 128, 2, stride=2)
            )
        self.lstm = nn.LSTM(self.get_output_shape(input_size,batch_size,input_channels), hidden_size, num_layers, 
                            batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def get_output_shape(self ,input_size,batch_size,input_channels):
        dim=batch_size,input_channels,input_size
        return self.resnet_half(torch.zeros(dim)).size(2)
    def forward(self,x):

        x=self.resnet_half(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
class mini_Residual_Conv_GRU(nn.Module):
    def __init__(self,input_size, batch_size,input_channels=12, inplanes=64, num_classes=9, GRU_hidden_size=128, GRU_num_layers=2):
        super().__init__()
        self.inplanes=inplanes
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )
        self.gru = nn.GRU(input_size = self.get_output_shape(input_size,batch_size,input_channels),
                        hidden_size = GRU_hidden_size, num_layers=GRU_num_layers, 
                        batch_first=True, dropout=0.5)
        self.fc = nn.Linear(GRU_hidden_size, num_classes)
        self.num_layers = GRU_num_layers
        self.hidden_size = GRU_hidden_size
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def get_output_shape(self ,input_size,batch_size,input_channels):
        dim=batch_size,input_channels,input_size
        return self.resnet_half(torch.zeros(dim)).size(2)    
    def forward(self,x):
        x=self.resnet_half(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)       
        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class Transformer(nn.Module):
    def __init__(self, num_classes,num_layers,model_dim=64):
        super(Transformer, self).__init__()
        self.in_proj=nn.Conv1d(12, 64, kernel_size=30, stride=2, padding=7, bias=False)
        #self.embedding = nn.Linear(12, model_dim)
        
        self.pos_encoder = PositionalEncoding(model_dim, dropout=0.1)
        encoder_layers = nn.TransformerEncoderLayer(model_dim, nhead=4, dim_feedforward=512, dropout=0.05)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(model_dim, num_classes)
        self.model_dim=model_dim
    def forward(self,x,src_mask=None):
        x =self.in_proj(x)
        x = x.permute(0,2,1)
        #x=self.embedding(x) * math.sqrt(self.model_dim)
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_mask)
        output = self.fc(x.mean(dim=1))
        return output

class Residual_ConvTransformer(nn.Module):
    def __init__(self,input_size, batch_size,input_channels=12, inplanes=64, num_classes=9, num_layers=2,model_dim=64):
        super().__init__()
        self.inplanes=inplanes
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            self._make_layer(BasicBlock1d, 128, 2, stride=2)
            )
        self.embedding = nn.Linear(128, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout=0.1)
        encoder_layers = nn.TransformerEncoderLayer(model_dim, nhead=32, dim_feedforward=512, dropout=0.05)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(model_dim, num_classes)
        self.model_dim = model_dim
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self,x,src_mask=None):
        x=self.resnet_half(x)
        x=x.permute(0,2,1)
        x=self.embedding(x) * math.sqrt(self.model_dim)
        
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_mask)
        output = self.fc(x.mean(dim=1))
        return output
class mini_Residual_ConvTransformer(nn.Module):
    def __init__(self,input_size, batch_size,input_channels=12, inplanes=64, num_classes=9, num_layers=2,model_dim=64):
        super().__init__()
        self.inplanes=inplanes
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            )
        self.embedding = nn.Linear(64, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout=0.1)
        encoder_layers = nn.TransformerEncoderLayer(model_dim, nhead=8, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(model_dim, num_classes)
        self.model_dim = model_dim
    def get_output_shape(self ,input_size,batch_size,input_channels):
        dim=batch_size,input_channels,input_size
        return self.resnet_half(torch.zeros(dim)).size(2)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self,x,src_mask=None):
        x=self.resnet_half(x)
        x=x.permute(0,2,1)
        x=self.embedding(x) * math.sqrt(self.model_dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_mask)
        output = self.fc(x.mean(dim=1))
        return output
class Residual_conv_retnet(nn.Module):
    def __init__(self,input_size, batch_size,input_channels=12, inplanes=64, num_classes=9, num_layers=2,hidden_dim=128,ffn_size=128, quantize=False):
        super().__init__()
        self.inplanes=inplanes
        self.resnet_half = nn.Sequential(
            nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(BasicBlock1d, 64, 2),
            self._make_layer(BasicBlock1d, 128, 2, stride=2)
            )
        self.layers = num_layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = 8
        self.double_v_dim = False
        self.v_dim = hidden_dim * 2 if self.double_v_dim else hidden_dim

        self.embedding = nn.Linear(self.get_output_shape(input_size,batch_size,input_channels), hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1)
        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, self.heads, self.double_v_dim,quantize)
            for _ in range(num_layers)
        ])       
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(hidden_dim, num_classes)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def get_output_shape(self ,input_size,batch_size,input_channels):
        dim=batch_size,input_channels,input_size
        return self.resnet_half(torch.zeros(dim)).size(2)  
    def forward(self,x):
    
        x=self.resnet_half(x)
        
        
        x=self.embedding(x) * math.sqrt(self.hidden_dim)
        x = self.pos_encoder(x)        
        
        for i in range(self.layers):
            x = self.retentions[i](self.layer_norms_1[i](x)) + x
           

        x=self.fc(x.mean(dim=1))

        return x



class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

time_list={}

class CLINet(nn.Module):
    def __init__(self, sequence_len, num_features=12,  num_classes=9):
        super(CLINet, self).__init__()
        self.num_features = num_features
        self.sequence_len = sequence_len
        self.num_classes = num_classes
        hidden_size = 128

        #self.conv0 = nn.Conv1d(num_features, num_features, kernel_size=15, stride=2, padding=7)

        self.conv1 = nn.Conv1d(num_features, 3, kernel_size=31, stride=5, padding=15)
        self.conv2 = nn.Conv1d(num_features, 3, kernel_size=36, stride=5, padding=16)
        self.conv3 = nn.Conv1d(num_features, 3, kernel_size=41, stride=5, padding=20)
        
        self.batch_norm1 = nn.BatchNorm1d(9)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=9, hidden_size=200, batch_first=True)
        
        # self.reshape1 = lambda x: x.reshape(x.size(0), hidden_size, self.sequence_len // 5, 1)
        
        self.inv1 = involution(channel=3, group_number=1, kernel_size=31, stride=5, reduction_ratio=2 )
        self.inv2 = involution(channel=3, group_number=1, kernel_size=36, stride=5, reduction_ratio=2 )
        self.inv3 = involution(channel=3, group_number=1, kernel_size=41, stride=5, reduction_ratio=2 )
        
        self.batch_norm2 = nn.BatchNorm2d(3)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(72000, 20)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, self.num_classes)

    def forward(self, x):
        
      
        # print(x.shape)

        #start_time = time.time()
        x11 = self.conv1(x)
        #print(x11.shape)
        #end_time = time.time()
        #print("conv1",end_time-start_time)
        x12 = self.conv2(x)
        x13 = self.conv3(x)
        x = torch.cat((x11, x12, x13), dim=1)
        
        x = self.batch_norm1(x)
        x = self.relu(x)
        


        x = x.permute(0, 2, 1)  # Change to (batch, seq_len, input_size) for LSTM
        #start_time = time.time()
        
        x, _ = self.lstm(x)

        #end_time = time.time()
        
        #print("lstm",end_time-start_time)
        #print(x.shape)
        
        x = x.unsqueeze(1)
        

        #start_time = time.time()
        x21  = self.inv1(x)

        #end_time = time.time()
        #print("inv1",end_time-start_time)
        x22  = self.inv2(x)
        x23  = self.inv3(x)

       
        # print(x21.shape,x22.shape,x23.shape)
        x = torch.cat((x21, x22, x23), dim=1)        

        x = self.batch_norm2(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        x = self.dropout1(x)
        #start_time = time.time()

        x = self.fc1(x)

        # end_time = time.time()
        # print(end_time-start_time)
        x = self.relu(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        
        return x


class Involution(nn.Module):
    def __init__(self, channel, group_number, kernel_size, stride, reduction_ratio, name=None):
        super(Involution, self).__init__()
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

        self.stride_layer = (
            nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0)
            if self.stride > 1 else nn.Identity()
        )

        self.kernel_gen = nn.Sequential(
            nn.Conv2d(1, channel // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(channel // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(channel // reduction_ratio, kernel_size * kernel_size * group_number, kernel_size=1)
        )

    def forward(self, x):
        batch_size, num_channels, ori_height, ori_width = x.size()
        height = ori_height // self.stride
        width = ori_width // self.stride
        
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)
        
        kernel = kernel.view(batch_size, self.group_number, self.kernel_size * self.kernel_size, height, width)
        kernel = kernel.permute(0, 3, 4, 2, 1)  # B, H, W, K*K, G
        kernel = kernel.unsqueeze(-2)
        #print(kernel.shape)
        
        
        patches = F.unfold(x, kernel_size=self.kernel_size, dilation=1, padding=self.kernel_size // 2, stride=self.stride)
        
        
        
        patches = patches.view(batch_size, num_channels // self.group_number, self.group_number, self.kernel_size * self.kernel_size, height, width)
        patches = patches.permute(0, 4, 5, 3, 1, 2)  # B, H, W, K*K, C//G, G
        #print(patches.shape)
        output = kernel * patches
        output = output.sum(dim=3)  # B, H, W, C
        
        output = output.squeeze(4)
        
        output =output.permute(0,3,1,2)
        #print(output.shape)
        return output, kernel
class involution(nn.Module):

    def __init__(self,channel, group_number, kernel_size, stride, reduction_ratio):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channel
        #reduction_ratio = 4
        #self.group_channels = 16
        #self.groups = self.channels // self.group_channels
        
        self.group_channels=group_number
        self.groups = 1
        self.kernel_gen = nn.Sequential(
            nn.Conv2d(1, channel // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(channel // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(channel // reduction_ratio, kernel_size * kernel_size * group_number, kernel_size=1)
        )
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, x):
        #start_time = time.time()
        weight = self.kernel_gen(x if self.stride == 1 else self.avgpool(x))

        #end_time = time.time()
        #print("weight",end_time-start_time)
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        
        #start_time = time.time()
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        #end_time = time.time()
        #print("unfold",end_time-start_time)

        #start_time = time.time()
        out = (weight * out).sum(dim=3)
        # end_time = time.time()
        # print("mm",end_time-start_time)
        
        out=out.view(b, 1, h, w)

        return out
################################### MLBF net
class MLBF_net(nn.Module):
    def __init__(self,nleads=12, num_classes=9):
        super(MLBF_net, self).__init__()
        self.nleads=nleads
        for i in range(nleads):
          setattr(self, "branch%d" % i, Branchnet(num_classes))
          
        self.attention = nn.Sequential(
           nn.Linear(in_features=12, out_features=12, bias=True),
           nn.Tanh(),
           nn.Linear(in_features=12, out_features=12, bias=False), 
           nn.Softmax(dim=1)
         )

    def  forward(self,x):
        branch_list=[]
        for i in range(self.nleads):
          branch_list.append(getattr(self, "branch%d" % i)(x[:,i,:].unsqueeze(1)))        
        x=torch.stack(branch_list, dim=2)
        score=self.attention(x)
        x=x * score
        x=torch.sum(x, dim=2)
        return x
    
class Branchnet(nn.Module):
    def __init__(self, num_classes=9):
        super(Branchnet, self).__init__()

        self.layer0 = nn.Sequential(
                nn.Conv1d(1,12,kernel_size=3,stride=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(12,12,kernel_size=3,stride=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(12,12,kernel_size=24,stride=2),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=0.2)
            )
        self.layer1 =  nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(12,12,kernel_size=3,stride=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(12,12,kernel_size=3,stride=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(12,12,kernel_size=24,stride=2),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=0.2)
            )
            for _  in range(3)
            ]
        )
        self.layer2 = nn.Sequential(
                nn.Conv1d(12,12,kernel_size=3,stride=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(12,12,kernel_size=3,stride=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv1d(12,12,kernel_size=48,stride=2),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=0.2)
            )
        self.biGRU= nn.GRU(input_size=12, hidden_size=12, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
           nn.Linear(in_features=24, out_features=24, bias=True),
           nn.Tanh(),
           nn.Linear(in_features=24, out_features=24, bias=False), 
           nn.Softmax(dim=1)
         )   
        self.fc = nn.Linear(in_features=24, out_features=num_classes)     
        
    def forward(self, x):
        x=self.layer0(x)
        for layer in self.layer1:
            x=layer(x)
        #B*12*913
        x=self.layer2(x)
        #B*12*431        
        h0 = torch.zeros(2, x.size(0), 12).to(x.device)
        x_0, _=self.biGRU(x.permute(0,2,1), h0)
        #B*431*24
        att_score=self.attention(x_0)
        x=x_0 * att_score
        x=torch.sum(x, dim=1)
        x=self.fc(x)
        
        return x

########################################## ResUDense
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(interChannels)
        self.conv2 = nn.Conv1d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, down=False):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm1d(nChannels)
        self.conv1 = nn.Conv1d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)
        self.down = down

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        if self.down:
            out = F.avg_pool1d(out, 2)
        return out


class ResidualUBlock(nn.Module):
    def __init__(self, out_ch, mid_ch, layers, downsampling=True):
        super(ResidualUBlock, self).__init__()
        self.downsample = downsampling  # Flag to decide if down-sampling is needed
        K = 9  # Kernel size
        P = (K - 1) // 2  # Padding calculation

        # Initial convolutional layer
        self.conv1 = nn.Conv1d(in_channels=out_ch,
                               out_channels=out_ch,
                               kernel_size=K,
                               padding=P,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.encoders = nn.ModuleList()  # Encoder layers
        self.decoders = nn.ModuleList()  # Decoder layers

        # Creating encoder-decoder blocks
        for idx in range(layers):
            # Encoder block definition
            if idx == 0:
                # First encoder has different input channels
                self.encoders.append(nn.Sequential(
                    nn.Conv1d(
                        in_channels=out_ch,
                        out_channels=mid_ch,
                        kernel_size=K,
                        stride=2,
                        padding=P,
                        bias=False
                    ),
                    nn.BatchNorm1d(mid_ch),
                    nn.LeakyReLU()
                ))
            else:
                # Subsequent encoders use mid_ch as input
                self.encoders.append(nn.Sequential(
                    nn.Conv1d(
                        in_channels=mid_ch,
                        out_channels=mid_ch,
                        kernel_size=K,
                        stride=2,
                        padding=P,
                        bias=False
                    ),
                    nn.BatchNorm1d(mid_ch),
                    nn.LeakyReLU()
                ))

            # Decoder block definition
            if idx == layers - 1:
                # Last decoder has different output channels
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels=mid_ch * 2,
                        out_channels=out_ch,
                        kernel_size=K,
                        stride=2,
                        padding=P,
                        output_padding=1,
                        bias=False
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.LeakyReLU()
                ))
            else:
                # Subsequent decoders output to mid_ch
                self.decoders.append(nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels=(mid_ch) * 2,
                        out_channels=mid_ch,
                        kernel_size=K,
                        stride=2,
                        padding=P,
                        output_padding=1,
                        bias=False
                    ),
                    nn.BatchNorm1d(mid_ch),
                    nn.LeakyReLU()
                ))

        # Bottleneck layer (center of U-Net)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                in_channels=mid_ch,
                out_channels=mid_ch,
                kernel_size=K,
                padding=P,
                bias=False
            ),
            nn.BatchNorm1d(mid_ch),
            nn.LeakyReLU()
        )

        # Down-sampling layers (if required)
        if self.downsample:
            self.idfunc_0 = nn.AvgPool1d(kernel_size=2, stride=2)
            self.idfunc_1 = nn.Conv1d(in_channels=out_ch,
                                      out_channels=out_ch,
                                      kernel_size=1,
                                      bias=False)

    def forward(self, x):
        x_in = F.leaky_relu(self.bn1(self.conv1(x)))

        out = x_in
        encoder_out = []
        for idx, layer in enumerate(self.encoders):
            # If output size is not divisible by 4, padding is added
            # if out.size(-1) % 4 != 0:
            #     out = functional.pad(out, [1, 0, 0, 0])                
            #     print("in")
            out = layer(out)
            
            encoder_out.append(out)

        out = self.bottleneck(out)
        
          
        for idx, layer in enumerate(self.decoders):           
            out = layer(torch.cat([out, encoder_out[-1 - idx]], dim=1))
            
            
        # Trim the output to match the size of x_in (input)
        out = out[..., :x_in.size(-1)]

        out += x_in

        # If down-sampling is required, apply down-sampling layers
        if self.downsample:
            out = self.idfunc_0(out)
            out = self.idfunc_1(out)

        return out


def _make_dense(nChannels, growthRate, nDenseBlocks, bottleneck):
    layers = []
    for i in range(int(nDenseBlocks)):
        if bottleneck:
            layers.append(Bottleneck(nChannels, growthRate))
        else:
            layers.append(SingleLayer(nChannels, growthRate))
        nChannels += growthRate
    return nn.Sequential(*layers)


class ResU_Dense(nn.Module):
    def __init__(self, nOUT, in_ch=12, out_ch=256, mid_ch=64):
        super(ResU_Dense, self).__init__()
        # Initial convolutional layer
        self.conv = nn.Conv1d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=15,
                              padding=7,
                              stride=2,
                              bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

        # Define Residual U-blocks
        self.rub_0 = ResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=6)
        self.rub_1 = ResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=5)
        self.rub_2 = ResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=4)
        self.rub_3 = ResidualUBlock(out_ch=out_ch, mid_ch=mid_ch, layers=3)

        # Parameters for dense blocks
        growthRate = 12
        reduction = 0.5
        nChannels = out_ch
        nDenseBlocks = 16

        # Define dense blocks and transitions
        self.dense1 = _make_dense(nChannels, growthRate=12, nDenseBlocks=nDenseBlocks, bottleneck=True)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = _make_dense(nChannels, growthRate=12, nDenseBlocks=nDenseBlocks, bottleneck=True)
        nChannels += nDenseBlocks * growthRate
        self.trans2 = Transition(nChannels, out_ch)

        # Multihead attention layer
        self.mha = nn.MultiheadAttention(out_ch, 8)
        # Max pooling layer to reduce dimensions
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

        # Fully connected layer
        self.fc_1 = nn.Linear(out_ch, nOUT)

    def forward(self, x, quantize=False):       
        
        if(quantize):
            x= torch.cat((x,torch.zeros(x.size(0),x.size(1),1000, dtype=torch.float16).to(x.device)) ,dim=2) #pad to division by4
        else:
            x= torch.cat((x,torch.zeros(x.size(0),x.size(1),1000).to(x.device)) ,dim=2) #pad to division by4
        
        
        x = F.leaky_relu(self.bn(self.conv(x)))

        
        # Pass through the residual U-blocks
        x = self.rub_0(x)
        x = self.rub_1(x)
        x = self.rub_2(x)
        x = self.rub_3(x)

        # Pass through the dense blocks and transitions
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))

        # Apply dropout for regularization
        x = F.dropout(x, p=0.5, training=self.training)

        x = x.permute(2, 0, 1)
        x, _ = self.mha(x, x, x)
        x = x.permute(1, 2, 0)

        # Reduce dimensions with pooling
        x = self.pool(x).squeeze(2)

        # Fully connected layer for final output
        x = self.fc_1(x)
        return x
############################################ SGB
class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1):
        super(DepthwiseSeparableConvolution, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv1d(nin, nin, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      groups=nin,
                      bias=False),
            nn.BatchNorm1d(nin),
            nn.GELU(),
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(nin, nout, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(nout),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)

        # Two different branches of ECA module
        y = self.conv(y).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, use_act=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv1d(in_channels, init_channels, kernel_size=kernel_size, stride=stride,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm1d(init_channels),
            nn.GELU() if use_act else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv1d(init_channels, new_channels, kernel_size=dw_size, stride=1,
                      padding=(dw_size - 1) // 2, groups=init_channels, bias=False),
            nn.BatchNorm1d(new_channels),
            nn.GELU() if use_act else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)

        return out[:, :self.out_channels, :]


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        N, C, L = x.size()
        g = self.groups

        return x.view(N, g, C // g, L).permute(0, 2, 1, 3).reshape(N, C, L)


class ShuffleGhostBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, hidden_ratio=2, use_se=False, shuffle=False):
        super(ShuffleGhostBottleneck, self).__init__()
        assert stride in [1, 2]
        hidden_channels = hidden_ratio * in_channels

        self.shuffle = ShuffleBlock(groups=2) if shuffle == 2 else nn.Sequential()

        self.conv = nn.Sequential(
            # pw
            GhostModule(in_channels, hidden_channels, kernel_size=1, use_act=True),

            # dw
            nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2, groups=hidden_channels, bias=False),
                nn.BatchNorm1d(hidden_channels),
            ) if stride == 2 else nn.Sequential(),

            # Squeeze-and-Excite
            ECALayer(hidden_channels) if use_se else nn.Sequential(),

            # pw-linear
            GhostModule(hidden_channels, out_channels, kernel_size=1, use_act=False),
        )

        if in_channels == out_channels and stride == 1:
            self.shortcut = lambda x: x
        else:
            self.shortcut = DepthwiseSeparableConvolution(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv(self.shuffle(x))

        return x + residual


class SGB(nn.Module):
    def __init__(self,  num_classes=9):
        super(SGB, self).__init__()
        # in_channels, out_channel, kernel_size, stride, hidden_ratio, use_se, shuffle
        cfgs = [
          [
            [32, 64, 3, 2, 2, 1, 1],
            [64, 64, 3, 1, 2, 1, 0],
            [64, 64, 3, 1, 2, 1, 0]
          ],
          [
            [64, 96, 3, 2, 2, 1, 1],
            [96, 96, 3, 1, 2, 1, 0],
            [96, 96, 3, 1, 2, 1, 0]
          ],
          [
            [96, 128, 3, 2, 2, 1, 1],
            [128, 128, 3, 1, 2, 1, 0],
          ]
        ]
        self.cfgs = cfgs
        num_stages = len(self.cfgs)
        in_proj_channel = self.cfgs[0][0][0]
        out_proj_channel = self.cfgs[-1][-1][0]

        self.in_proj = nn.Sequential(
            nn.Conv1d(12, in_proj_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(in_proj_channel),
            nn.LeakyReLU(inplace=True),
        )

        layers = []
        for i in range(num_stages):
            for in_c, out_c, k, s, r, use_se, shuffle in self.cfgs[i]:
                layers.append(ShuffleGhostBottleneck(in_c, out_c, k, s, r, use_se, shuffle))
        self.layers = nn.Sequential(*layers)

        self.out_proj = nn.Sequential(
            nn.Conv1d(out_proj_channel, 1024, 1, 1, 0, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.layers(x)
        x = self.out_proj(x)
        x = self.gap(x)
        x = self.classifier(x)

        return x
#######################################################cpsc_champion

def dot_product(x, kernel):

    print(x.shape,kernel.unsqueeze(0).shape)
    result=torch.matmul(x, kernel.unsqueeze(0))
    return torch.squeeze(result, -1)

class AttentionWithContext(nn.Module):
    def __init__(self, input_dim, bias=True):
        super(AttentionWithContext, self).__init__()
        self.bias = bias
        self.W = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.u = nn.Parameter(torch.Tensor(input_dim, input_dim))
        if self.bias:
            self.b = nn.Parameter(torch.Tensor(input_dim))
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)
        if self.bias:
            nn.init.zeros_(self.b)

    def forward(self, x, mask=None):
        print(x.shape, self.W.shape)
        uit = dot_product(x, self.W)
        if self.bias:
            uit += self.b
        uit = torch.tanh(uit)
        ait = dot_product(uit, self.u)
        a = torch.exp(ait)
        if mask is not None:
            a = a * mask.float()
        a = a / (torch.sum(a, dim=1, keepdim=True) + 1e-10)
        a = a.unsqueeze(-1)
        weighted_input = x * a
        return torch.sum(weighted_input, dim=1)


class cpsc_champion(nn.Module):
    def __init__(self, seq_len,num_classes):
        super(cpsc_champion, self).__init__()
       
       
        self.convblock = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(12, 12, 3, padding='same'),
                nn.LeakyReLU(negative_slope=0.3),
                nn.Conv1d(12, 12, 3, padding='same'),
                nn.LeakyReLU(negative_slope=0.3),
                nn.Conv1d(12, 12, 24,  stride=2,padding=1),
                nn.LeakyReLU(negative_slope=0.3),
                nn.Dropout(0.2)
            )
            for _ in range(4)
        ] 
        )
        self.convblock2 = nn.Sequential(
            nn.Conv1d(12, 12, 3, padding='same'),
                nn.LeakyReLU(negative_slope=0.3),
                nn.Conv1d(12, 12, 3, padding='same'),
                nn.LeakyReLU(negative_slope=0.3),
                nn.Conv1d(12, 12, 48, stride=2,padding=1),
                nn.LeakyReLU(negative_slope=0.3),
                nn.Dropout(0.2)
        )
        
        self.bi_gru = nn.GRU(12, 12, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(24,24,bias=True),
            nn.Tanh(),
            nn.Linear(24,24,bias=False),
            nn.Softmax(dim=1)
        )       
        
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(24)
        self.fc = nn.Linear(24, num_classes)
        
    def forward(self, x):
        a=x
        for layer in self.convblock:
            x = layer(x)
        x=self.convblock2(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bi_gru(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        x = self.dropout(x)

        score = self.attention(x)
        
        x = score*x
        
        x=torch.sum(x, dim=1)
        if(a.size(0)!=1):
            x = self.batch_norm(x)
        x = F.leaky_relu(x, negative_slope=0.3)
        x = self.dropout(x)
        x = self.fc(x)
        return x #torch.sigmoid(x)




#######################################################
def resnet18(**kwargs):
    model = ResNet1d(BasicBlock1d, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 4, 6, 3], **kwargs)
    return model

def resnet10(**kwargs):
    model = ResNet1d(BasicBlock1d, [1, 1, 1, 1], **kwargs)
    return model

def resnet12(**kwargs):
    model = ResNet1d(BasicBlock1d, [1, 1, 2, 1], **kwargs)
    return model

def resnet14(**kwargs):
    model = ResNet1d(BasicBlock1d, [1, 2, 2, 1], **kwargs)
    return model

def resnet22(**kwargs):
    model = ResNet1d(BasicBlock1d, [2, 3, 3, 2], **kwargs)
    return model

def resnet26(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 3, 3, 3], **kwargs)
    return model

def resnet28(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 3, 4, 3], **kwargs)
    return model

def resnet30(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 4, 4, 3], **kwargs)
    return model

def resnet40(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 6, 7, 3], **kwargs)
    return model

def resnet38(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 6, 6, 3], **kwargs)
    return model

def resnet42(**kwargs):
    model = ResNet1d(BasicBlock1d, [3, 7, 7, 3], **kwargs)
    return model

#######################################
# Mamba
########################################
class Mamba(nn.Module):
    """
    Mamba, linear-time sequence modeling with selective state spaces O(L)
    Paper link: https://arxiv.org/abs/2312.00752
    Implementation refernce: https://github.com/johnma2006/mamba-minimal/
    """

    def __init__(self, d_model, expand, enc_in, c_out, d_conv, d_ff, e_layers=2, dropout=0.1):
        super(Mamba, self).__init__()
        # d_model : dimension of model
        # expand : expansion factor for Mamba
        # enc_in : encoder input size
        # e_layers : encoder layers 
        # c_out : output_size
        # d_conv : conv kernel size
        # d_ff : dimension of fcn
        embed = 'timeF' #[timeF, fixed, learned]
        freq='s'
        sequence_length=15000


        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        self.token_embedding = TokenEmbedding(sequence_length, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model,max_len=sequence_length)
        #DataEmbedding(enc_in, d_model, embed, freq, dropout)

        self.input_layer = nn.Conv1d(12, d_model, kernel_size=15, stride=2, padding=7, bias=False)

        self.layers = nn.ModuleList([ResidualBlock( d_model,  d_conv, d_ff, self.d_inner, self.dt_rank) for _ in range(e_layers)])
        self.norm = RMSNorm(d_model)

        #self.fc = nn.Linear(d_model, c_out, bias=False)
        self.out_layer=nn.Linear(d_model, c_out, bias=False)
    # # def short_term_forecast(self, x_enc, x_mark_enc):
    # def forecast(self, x_enc, x_mark_enc):
    #     mean_enc = x_enc.mean(1, keepdim=True).detach()
    #     x_enc = x_enc - mean_enc
    #     std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    #     x_enc = x_enc / std_enc

    #     x = self.embedding(x_enc, x_mark_enc)
    #     for layer in self.layers:
    #         x = layer(x)

    #     x = self.norm(x)
    #     x_out = self.out_layer(x)

    #     x_out = x_out * std_enc + mean_enc
    #     return x_out

    def forward(self, x):

        
        x=self.token_embedding(x)
        #print(x.shape)
       

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x_out = self.out_layer(x.mean(dim=1))
        #print("vvadav",x_out.shape)
        return x_out
class ResidualBlock(nn.Module):
    def __init__(self, d_model,  d_conv, d_ff, d_inner, dt_rank):
        super(ResidualBlock, self).__init__()
        
        self.mixer = MambaBlock( d_model, d_conv, d_ff, d_inner, dt_rank)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_conv, d_ff, d_inner, dt_rank):
        super(MambaBlock, self).__init__()
        self.d_inner = d_inner
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels = self.d_inner,
            out_channels = self.d_inner,
            bias = True,
            kernel_size = d_conv,
            padding = d_conv - 1,
            groups = self.d_inner
        )

        # takes in x and outputs the input-specific delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_ff * 2, bias=False)

        # projects delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, d_ff + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Figure 3 in Section 3.4 in the paper
        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x) # [B, L, 2 * d_inner]
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)

        output = self.out_proj(y)
        return output


    def ssm(self, x):
        """
        Algorithm 2 in Section 3.2 in the paper
        """
        
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float()) # [d_in, n]
        D = self.D.float() # [d_in]

        x_dbl = self.x_proj(x) # [B, L, d_rank + 2 * d_ff]
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1) # delta: [B, L, d_rank]; B, C: [B, L, n]
        delta = F.softplus(self.dt_proj(delta)) # [B, L, d_in]
        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n")) # A is discretized using zero-order hold (ZOH) discretization
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n") # B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors: "A is the more important term and the performance doesn't change much with the simplification on B"

        # selective scan, sequential instead of parallel
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)

        y = torch.stack(ys, dim=1) # [B, L, d_in]
        y = y + u * D

        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

#####################################
#retnet
#####################################
class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size=None, double_v_dim=False,quantize=False):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        self.quantize=True if quantize else False

        self.hidden_size = hidden_size
        if head_size is None:
            head_size = hidden_size
    
        self.head_size = head_size

        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        
        self.xpos = XPOS(head_size)

    def forward(self, X):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length).to(self.W_Q.device)

        Q = (X @ self.W_Q)
        K = (X @ self.W_K)

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        V = X @ self.W_V
        if(self.quantize):
            ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0).half()
        else:
            ret = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)
        
        return ret @ V
        
    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, 1, hidden_size)
        s_n_1: (batch_size, hidden_size, v_dim)
        """

        Q = (x_n @ self.W_Q)
        K = (x_n @ self.W_K)

        Q = self.xpos(Q, n+1)
        K = self.xpos(K, n+1, downscale=True)

        V = x_n @ self.W_V

        # K: (batch_size, 1, hidden_size)
        # V: (batch_size, 1, v_dim)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)
        
        return (Q @ s_n), s_n
    
    def forward_chunkwise(self, x_i, r_i_1, i):
        """
        Chunkwise representation of the retention mechanism.
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1: (batch_size, hidden_size, v_dim)
        """
        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size)

        Q = (x_i @ self.W_Q)
        K = (x_i @ self.W_K)

        Q = self.xpos(Q, i * chunk_size)
        K = self.xpos(K, i * chunk_size, downscale=True)

        V = x_i @ self.W_V
        
        r_i =(K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V
        
        #e[i,j] = gamma ** (i+1)
        e = torch.zeros(batch, chunk_size, 1)
        
        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)
        
        cross_chunk = (Q @ r_i_1) * e
        
        return inner_chunk + cross_chunk, r_i

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D
    


class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim=False,quantize=False):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size
        
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim, quantize) for gamma in self.gammas
        ])

    def forward(self, X):
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to X
        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X))
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(X @ self.W_G) * Y) @ self.W_O
    
    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        recurrent representation of the multi-scale retention mechanism
        x_n: (batch_size, 1, hidden_size)
        s_n_1s: (batch_size, heads, head_size, head_size)

        """
    
        # apply each individual retention mechanism to a slice of X
        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(
                x_n[:, :, :], s_n_1s[i], n
                )
            Y.append(y)
            s_ns.append(s_n)
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        
        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        chunkwise representation of the multi-scale retention mechanism
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1s: (batch_size, heads, head_size, head_size)
        """
        batch, chunk_size, _ = x_i.shape

        # apply each individual retention mechanism to a slice of X
        Y = []
        r_is = []
        for j in range(self.heads):
            y, r_i = self.retentions[j].forward_chunkwise(
                x_i[:, :, :], r_i_1s[j], i
                )
            Y.append(y)
            r_is.append(r_i)
        
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is

class RetNet(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, sequence_length, features, num_classes, double_v_dim=False):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim
        
        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])
       
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        # Initial linear transformation

        # print("fwef3")
        # self.initial_transform = nn.Linear(sequence_length,  hidden_dim)
        # print("fwef")
        self.token_embedding = TokenEmbedding(sequence_length, hidden_dim)
        #self.input_layer = nn.Conv1d(features, hidden_dim, kernel_size=15, stride=2, padding=7, bias=False)
        # # Output classifier
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sequence_length=sequence_length
        self.features=features
        

    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        batch_size, sequence_length, features = X.shape

        # assert(self.sequence_length*self.features  % self.hidden_dim ==0, "sequence_len*features%hidden_dim=0")
       
        X = self.token_embedding(X)             
             
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X
           
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y

        X=self.fc(X.mean(dim=1))
        
        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        s_ns = []
        for i in range(self.layers):
            # list index out of range
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n
        
        return x_n, s_ns
    
    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        X: (batch_size, sequence_length, hidden_size)
        r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        r_is = []
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
            y_i = o_i + x_i
            r_is.append(r_i)
            x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i
        
        return x_i, r_is