require('torch')
require('nn')
require('parse')
require('features')
require('sparse_lookup')
require('PairwiseDot')
require('optim')

-- This is the standard linear feature model.
function make_model()
   local scorer = nn.Sequential()
   -- This is just a lookup table.
   lookup = nn.SparseUpdateLookupTable(featureLimit, 1)
   lookup.weight:zero()
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
   scorer:add(par)
   -- Modifier
   par:add(nn.LookupTable(dict.num_words, 50))
   -- Head
   par:add(nn.LookupTable(dict.num_words, 50))
   scorer:add(nn.PairwiseDotProduct())
   return scorer
end

function make_model_linear_dot(dict)
   local scorer = nn.Sequential()
   local par = nn.ParallelTable()
   score:add(par)
   -- Modifier
   local mod_lookup = nn.Sequential()
   mod_lookup:add(nn.LookupTable(dict.num_words, 50))
   mod_lookup:add(nn.Linear(50,50))
   par:add(mod_lookup)
   -- Head
   local head_lookup = nn.Sequential()
   head_lookup:add(nn.LookupTable(dict.num_words, 50))
   head_lookup:add(nn.Linear(50,50))
   par:add(head_lookup)

   score:add(nn.DotProduct())
   return scorer
end


function make_example(sent, dict)
   local input = torch.zeros(2, #sent+1):long()
   local target = torch.zeros(#sent+1):double()

   -- The root word.
   input[1][1] = 1
   input[2][1] = 1
   target[1] = 1
   for j = 1, #sent do
      input[1][j+1] = dict.symbol_to_index[sent[j].word] or 1
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
   local scorer2 = make_model_dot(dict)
   local combine = nn.CAddTable()

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

         if #sent < 100 then
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

            local out = arc_scores
            -- For Embeddings
            -- local HEAD = parts:t()[1]
            -- local MOD = parts:t()[2]
            -- local head_words = input[1]:index(1, HEAD):long()
            -- local mod_words = input[1]:index(1, MOD):long()

            -- local arc_combine = scorer2:forward({head_words, mod_words})
            -- local out = combine:forward({arc_scores, arc_combine})

            l2 = os.clock()
            -- print("features", l2 - l1)

            local loss = parser:forward(out, target)
            total_loss = total_loss + loss
            l3 = os.clock()
            -- print("inference", l3 - l2)

            local deriv = parser:backward(out, target)

            -- Update (SGD)
            scorer:zeroGradParameters()
            -- local d = combine:backward({arc_scores, arc_combine}, deriv)
            scorer:backward(features:t(), deriv)
            scorer:updateParameters(rate)
            -- scorer2:backward({head_words, mod_words}, d[2])
            -- scorer2:updateParameters(rate)
            l4 = os.clock()
            -- print("update", l4 - l3)

            -- Log
            total_sentences = total_sentences + 1
            if total_sentences % 100 == 0 then
               print(total_sentences, total_loss / total_sentences)
            end
         end
      end
      torch.save("/tmp/model", scorer)
      -- torch.save("/tmp/mix_model.2", scorer2)
      print("loss", total_loss / total_sentences)
   end
end

function test()

   -- Read sentence
   local _, dict =
      read_conll("/home/srush/data/wsj/converted", 40000)

   local sentences, _ =
      read_conll("/home/srush/Projects/PhraseDep/corpora/proj.full.dev.tbttagged.predict", 40000)

   -- Standard sparse scorer.
   local scorer = torch.load("/tmp/model")
   local parser = nn.Parser()

   local offsets = feature_templates(dict)
   for i = 1, #sentences do
      local sent = sentences[i]

      l0 = os.clock()
      local input, target = make_example(sent, dict)
      la = os.clock()
      -- print("make", la - l0)

      -- A little bit of caching to speed up features.
      parts = generate_parts(target:size(1))
      features = torch.ones(#offsets, parts:size(1)):long()
      features_mat(input, parts, offsets, features)

      l1 = os.clock()
      -- print("parts", l1 - la)

      local arc_scores = scorer:forward(features:t())

      l2 = os.clock()
      -- print("features", l2 - l1)

      local loss = parser:forward(arc_scores, target)
      for i = 2, parser.argmax:size(1) do
         print(string.format("%d\t%s\t_\t%s\t%s\t_\t%d\t_",
                             i-1, sent[i-1].word,  sent[i-1].tag, sent[i-1].tag, parser.argmax[i]-1))
      end
      print("")
   end
end

-- main()
test()
