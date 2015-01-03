require('torch')
require('nn')
require('libparse')

include('Parser.lua')

nn.Parser():forward(torch.Tensor({{1,2},{1,2}}))
