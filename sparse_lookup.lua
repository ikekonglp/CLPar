
local SparseUpdateLookupTable, parent = torch.class('nn.SparseUpdateLookupTable', 'nn.LookupTable')

function SparseUpdateLookupTable:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.nBackward = self.nBackward + 1
      for i=1,input:size(1) do
         local k = input[i]
         self.inputs[k] = (self.inputs[k] or 0) + 1
         self.gradWeight:select(1, k):add(scale, gradOutput:select(1, i))
      end
   elseif input:dim() == 2 then
      self.nBackward = self.nBackward + input:size(1)
      for i=1,input:size(1) do
         local input = input:select(1, i)
         local gradOutput = gradOutput:select(1, i)
         if gradOutput:sum() ~= 0 then
            for j=1,input:size(1) do
               local k = input[j]
               self.inputs[k] = (self.inputs[k] or 0) + 1
               self.gradWeight:select(1, k):add(scale, gradOutput:select(1, j))
            end
         end
      end
   end
end
