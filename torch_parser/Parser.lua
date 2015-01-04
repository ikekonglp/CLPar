require 'nn'

local Parser, parent = torch.class('nn.Parser', 'nn.Criterion')

function Parser:__init()
   self.argmax = torch.Tensor(100)
   self.output = torch.Tensor(1)
   self.gradInput = torch.Tensor(1)
   parent.__init(self)
end

function Parser:updateOutput(input, target)
   local n = target:size(1)
   input.parse.Parser_updateOutput(self, input, target)
   -- Move all the indices up 1 (1-based lua).
   self.argmax = self.argmax + 1
   local hamming = 0
   for i = 2, target:size(1) do
      if self.argmax[i] ~= target[i] then
         hamming = hamming + 1
      end
   end
   self.output = hamming
   return self.output / n
end

function Parser:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   local n = target:size(1)
   self.gradInput:zero()
   for i = 2, target:size(1) do
      if self.argmax[i] ~= target[i] then
         self.gradInput[self:to_pos(n, target[i], i)] = -1
         self.gradInput[self:to_pos(n, self.argmax[i], i)] = 1
      end
   end
   return self.gradInput
end

function Parser:score(input, target)
   local n = target:size(1)
   local score = 0
   for i = 2, target:size(1) do
      score = score + input[self:to_pos(n, target[i], i)]
   end
   return score
end

function Parser:to_pos(n, head, mod)
   local index = n * (head-1) + (mod-1) + 1
   return index
end

return Parser
