
require 'nn'

local Parser, parent = torch.class('nn.Parser', 'nn.Module')

function Parser:__init()
   parent.__init(self)
end

function Parser:updateOutput(input)
   input.nn.Parser_updateOutput(self, input)
   return self.output
end

function Parser:updateGradInput(input, gradOutput)
   input.nn.Parser_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

return Parser
