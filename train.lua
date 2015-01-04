require('torch')
require('nn')
require('parse')
require('features')

function make_model()
   local scorer = nn.Sequential()
   scorer:add(nn.LookupTable(1000000, 1))
   scorer:add(nn.Sum(2))
   return scorer
end


function make_example(sent, dict)
   local input = torch.zeros(2, #sent+1):long()
   local target = torch.zeros(#sent+1):double()

   -- The root word.
   input[1][1] = 1
   input[1][2] = 1
   target[1] = 1
   for j = 1, #sent do
      input[1][j+1] = dict.symbol_to_index[sent[j].word]
      input[2][j+1] = dict.tag_to_index[sent[j].tag]
      target[j+1] = sent[j].head + 1
   end
   return input, target
end

function main()
   -- Read sentence
   local sentences, dict =
      read_conll("/home/srush/data/wsj/converted", 1000)

   -- Standard sparse scorer.
   local scorer = make_model()
   local parser = nn.Parser()
   local rate = 1

   local offsets = feature_templates(dict)
   for epochs = 1, 10 do
      print("Epoch", epochs)
      local total_sentences = 0
      local total_loss = 0

      for i = 1, #sentences do
         local sent = sentences[i]
         if #sent < 20 then
            local input, target = make_example(sent, dict)
            local parts = generate_parts(target:size(1))
            local features = features(input, parts, offsets)
            local arc_scores = scorer:forward(features)

            local loss = parser:forward(arc_scores, target)
            total_loss = total_loss + loss
            local deriv = parser:backward(arc_scores, target)

            -- Update (SGD)
            scorer:zeroGradParameters()
            scorer:backward(features, deriv)
            scorer:updateParameters(rate)

            -- Log
            total_sentences = total_sentences + 1
            if total_sentences % 100 == 0 then
               print(total_sentences, total_loss / total_sentences)
            end
         end
      end
      print("loss", total_loss / total_sentences)
   end
end

main()
