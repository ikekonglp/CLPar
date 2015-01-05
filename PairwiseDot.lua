local PairwiseDotProduct, parent = torch.class('nn.PairwiseDotProduct', 'nn.Module')

function PairwiseDotProduct:__init()
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
   self.output=torch.Tensor(1)
end

function PairwiseDotProduct:updateOutput(input, y)
   self.output:resize(input[1]:size(1))
   for i = 1, input[1]:size(1) do
      self.output[i] = input[1][i]:dot(input[2][i])
   end
   return self.output
end

function PairwiseDotProduct:updateGradInput(input, gradOutput)
   local v1 = input[1]
   local v2 = input[2]
   local gw1=self.gradInput[1];
   local gw2=self.gradInput[2];
   gw1:resizeAs(v1)
   gw2:resizeAs(v2)

   gw1:copy( v2)
   gw1:mul(gradOutput[1])

   gw2:copy( v1)
   gw2:mul(gradOutput[1])

   return self.gradInput
end

function PairwiseDotProduct:type(type)
   for i, tensor in ipairs(self.gradInput) do
      self.gradInput[i] = tensor:type(type)
   end
   return parent.type(self, type)
end
