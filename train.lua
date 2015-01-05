require('torch')
require('nn')
require('parse')
require('features')
require('sparse_lookup')

-- This is the standard linear feature model.
function make_model()
   local scorer = nn.Sequential()
   -- This is just a lookup table.
   scorer:add(nn.SparseUpdateLookupTable(featureLimit, 1))
   scorer:add(nn.Sum(2))
   return scorer
end

-- This is a sample of another non-linear scorer
-- where the score is dot product of the embeddings of head and modifier.
-- Instead of passing in features we would do forward({head, mod}).
function make_model_dot(dict)
   local scorer = nn.Sequential()
   local par = nn.ParallelTable()
   score:add(par)
   -- Modifier
   par:add(nn.LookupTable(dict.num_words))
   -- Head
   par:add(nn.LookupTable(dict.num_words))
   score:add(nn.DotProduct())
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
      read_conll("/home/srush/data/wsj/converted", 40000)

   -- Standard sparse scorer.
   local scorer = make_model()
   local parser = nn.Parser()
   local rate = 1

   local offsets = feature_templates(dict)
   local parts_cache = {}
   for epochs = 1, 10 do
      print("Epoch", epochs)
      local total_sentences = 0
      local total_loss = 0

      for i = 1, #sentences do
         local sent = sentences[i]

         if #sent < 50 then
            l0 = os.clock()
            local input, target = make_example(sent, dict)
            la = os.clock()
            -- print("make", la - l0)

            -- A little bit of caching to speed up features.
            local parts
            if parts_cache[target:size(1)] then
               parts = parts_cache[target:size(1)].parts
               features = parts_cache[target:size(1)].features
            else
               obj = {}
               obj.parts = generate_parts(target:size(1))
               obj.features = torch.ones(#offsets, obj.parts:size(1)):long()
               parts_cache[target:size(1)] = obj
               parts = obj.parts
               features = obj.features
            end
            features_mat(input, parts, offsets, features)

            l1 = os.clock()
            -- print("parts", l1 - la)

            local arc_scores = scorer:forward(features:t())

            l2 = os.clock()
            -- print("features", l2 - l1)

            local loss = parser:forward(arc_scores, target)
            total_loss = total_loss + loss
            l3 = os.clock()
            -- print("inference", l3 - l2)

            local deriv = parser:backward(arc_scores, target)

            -- Update (SGD)
            scorer:zeroGradParameters()
            scorer:backward(features:t(), deriv)
            scorer:updateParameters(rate)
            l4 = os.clock()
            -- print("update", l4 - l3)

            -- Log
            total_sentences = total_sentences + 1
            if total_sentences % 100 == 0 then
               print(total_sentences, total_loss / total_sentences)
            end
         end
      end
      -- torch.save("/tmp/model", scorer)
      print("loss", total_loss / total_sentences)
   end
end

main()
