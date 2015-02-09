require('torch')
require('nn')
require('parse')
require('features')
require('sparse_lookup')
require('PairwiseDot')
require('optim')

local word_vec_length = 50

-- This is the standard linear feature model.
function make_model()
   local scorer = nn.Sequential()
   -- This is just a lookup table.
   lookup = nn.SparseUpdateLookupTable(featureLimit, 1)
   lookup.weight:zero()
   scorer:add(lookup)
   -- 2 just means sum the column (1 means sum the row)
   scorer:add(nn.Sum(2))
   return scorer
end

-- This is a sample of another non-linear scorer
-- where the score is dot product of the embeddings of head and modifier.
-- Instead of passing in features we would do forward({head, mod}).
function make_model_embedding(dict)
   local use_embed = true

   local scorer = nn.Sequential()
   local par = nn.ParallelTable()

   local mod_lookup = nn.Sequential()
   local head_lookup = nn.Sequential()

   local mod_par = nn.ParallelTable()
   local head_par = nn.ParallelTable()

   local lwt1 = nn.LookupTable(dict.num_words, word_vec_length)
   local lwt2 = nn.LookupTable(dict.num_words, word_vec_length)

   local post1 = nn.LookupTable(dict.num_tags, 20)
   local post2 = nn.LookupTable(dict.num_tags, 20)

   if use_embed then
      local word2emd = prepare()

      for word, value in pairs(word2emd) do
         -- print (word2emd[word])
         lwt1.weight[dict.symbol_to_index[word]] = word2emd[word];
         lwt2.weight[dict.symbol_to_index[word]] = word2emd[word];
      end
   end

   mod_par:add(lwt1)
   mod_par:add(post1)
   mod_lookup:add(mod_par)
   mod_lookup:add(nn.JoinTable(2))

   mod_lookup:add(nn.Linear((word_vec_length + 20), 20))
   mod_lookup:add(nn.Tanh())

   head_par:add(lwt2)
   head_par:add(post2)
   head_lookup:add(head_par)
   head_lookup:add(nn.JoinTable(2))

   head_lookup:add(nn.Linear((word_vec_length + 20), 20))
   head_lookup:add(nn.Tanh())

   par:add(mod_lookup)
   par:add(head_lookup)

   scorer:add(par)
   scorer:add(nn.JoinTable(2))

   -- scorer:add(nn.PairwiseDotProduct())

   -- scorer:add(nn.Linear(40, 40))
   -- scorer:add(nn.HardTanh())
   scorer:add(nn.Linear(40,1))
   scorer:add(nn.HardTanh())

   -- scorer:add(nn.JoinTable(2))

   
   -- ll1 = nn.Linear((word_vec_length + 20) * 2, (word_vec_length+20) * 3)
   -- hth = nn.HardTanh()
   -- ll2 = nn.Linear((word_vec_length + 20) * 3, 1)

   -- scorer:add(ll1)
   -- scorer:add(hth)
   -- scorer:add(ll2)

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
   local gold = torch.zeros( ((#sent+1) * (#sent+1)) ):double()
   local temp = torch.ones( ((#sent+1) * (#sent+1)) ):double()
   gold = gold - temp

   -- The root word.
   -- root word and pos index -> 1
   input[1][1] = 1
   input[2][1] = 1
   target[1] = 1

   gold[1] = 1

   for j = 1, #sent do
      input[1][j+1] = dict.symbol_to_index[sent[j].word] or 1

      input[2][j+1] = dict.tag_to_index[sent[j].tag]
      --print (dict)
      target[j+1] = sent[j].head + 1
      -- j's head is sent[j].head
      -- in 1 index system, this is saying
      local position = ( sent[j].head * (#sent + 1) ) + (j + 1)
      gold[position] = 1
   end

   return input, target, gold
end

function main()
   print ("start training")
   -- Read sentence
   local sentences, dict =
      -- read_conll("/home/srush/data/wsj/converted", 40000)
      read_conll("/home/lingpenk/Data/PTB/PTB_YM/train_final_finepos", 40000)

   -- Standard sparse scorer.
   
   -- try different scorer
   -- local scorer = make_model()
   local scorer2 = make_model_embedding(dict)

   -- local combine = nn.CAddTable()

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
            -- input is two row refers to word and coarse pos index
            -- target is the gold output (+1 indexing)

            local input, target, gold = make_example(sent, dict)
            -- print (target)

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

            -- features:
            --         . . . .  feature_i . . . .
            --  .
            --  .
            -- part_i
            --  .
            --  .

            -- features_mat(input, parts, offsets, features)

            -- fill the feature matrix with real features for each part

            l1 = os.clock()
            -- print("parts", l1 - la)

            -- forward the feature tensor as input
            -- the size of the tensor is (n^2) * feature_lenght_represented by long
            -- local arc_scores = scorer:forward(features:t())

            -- local out = arc_scores

            -- For Embeddings
            local HEAD = parts:t()[1]
            local MOD = parts:t()[2]
            local head_words = input[1]:index(1, HEAD):long()
            local mod_words = input[1]:index(1, MOD):long()
            local head_pos = input[2]:index(1, HEAD):long()
            local mod_pos = input[2]:index(1, MOD):long()
            -- print (head_pos)

            criterion = nn.MSECriterion()


            -- local out = scorer2:forward({head_words, head_pos, mod_words, mod_pos})
            torch.randn(10,2)
            local out = scorer2:forward({{head_words, head_pos}, {mod_words, mod_pos}})

            criterion:forward(scorer2:forward({{head_words, head_pos}, {mod_words, mod_pos}}), gold)

            scorer2:zeroGradParameters()
            scorer2:backward({{head_words, head_pos}, {mod_words, mod_pos}}, criterion:backward(scorer2.output, gold))
            scorer2:updateParameters(0.4)

            -- print (out,gold)

            -- criterion:forward(out, gold)
            
            -- -- only for compute the loss of the parser
            -- local loss = parser:forward(out, target)

            -- scorer2:zeroGradParameters()
            -- scorer2:backward({head_words, mod_words}, criterion:backward(out, gold))
            -- scorer2:updateParameters(0.01)



            -- local deriv = parser:backward(out, target)
            -- scorer2:zeroGradParameters()

            -- criterion = nn.MSECriterion()
            -- local d = scorer2.backward(out, derive)

            -- l2 = os.clock()
            -- -- print("features", l2 - l1)

            -- -- the parser here sounds like a module in nn, but it's acutally running a black box inside
            -- -- which gives the decoded result
            local loss = parser:forward(out, target)
            -- local loss = parser:forward(gold, target)
            total_loss = total_loss + loss
            -- l3 = os.clock()
            -- -- print("inference", l3 - l2)

            -- local deriv = parser:backward(out, target)

            -- -- perform the backward and update the thing as usual
            -- -- Update (SGD)
            -- -- scorer:zeroGradParameters()
            -- -- local d = combine:backward({arc_scores, arc_combine}, deriv)
            -- local d = scorer2.backward(out, derive)
            -- -- scorer:backward(features:t(), deriv)
            -- -- scorer:updateParameters(rate)
            
            -- -- scorer2:backward({head_words, mod_words}, d[2])
            -- scorer2:updateParameters(rate)
            -- l4 = os.clock()
            -- -- print("update", l4 - l3)

            -- Log
            total_sentences = total_sentences + 1
            if total_sentences % 100 == 0 then
               print(total_sentences, total_loss / total_sentences)
            end
         end
         torch.save("ptbada.model.epoch" .. epochs, scorer2)
      end
      -- torch.save("/tmp/model", scorer)
      torch.save("ptbada.model", scorer2)
      -- torch.save("/tmp/mix_model.2", scorer2)
      print("loss", total_loss / total_sentences)
   end
end

function test()

   -- Read sentence
   local _, dict =
      -- read_conll("/home/srush/data/wsj/converted", 40000)
      read_conll("/home/lingpenk/Data/PTB/PTB_YM/train_final_finepos", 40000)

   local sentences, _ =
      -- read_conll("/home/srush/Projects/PhraseDep/corpora/proj.full.dev.tbttagged.predict", 40000)
      read_conll("/home/lingpenk/Data/PTB/PTB_YM/dev_stg_finepos", 40000)
      -- read_conll("/home/lingpenk/Data/PTB/PTB_YM/train_final_finepos", 100)

   -- Standard sparse scorer.
   local scorer2 = torch.load("ptbada.model")
   local parser = nn.Parser()

   local offsets = feature_templates(dict)
   for i = 1, #sentences do
      local sent = sentences[i]

      l0 = os.clock()
      local input, target, gold = make_example(sent, dict)
      la = os.clock()
      -- print("make", la - l0)

      -- A little bit of caching to speed up features.
      parts = generate_parts(target:size(1))

      -- For Embeddings
      local HEAD = parts:t()[1]
      local MOD = parts:t()[2]
      local head_words = input[1]:index(1, HEAD):long()
      local mod_words = input[1]:index(1, MOD):long()
      local head_pos = input[2]:index(1, HEAD):long()
      local mod_pos = input[2]:index(1, MOD):long()


      -- features = torch.ones(#offsets, parts:size(1)):long()
      -- features_mat(input, parts, offsets, features)

      l1 = os.clock()
      -- print("parts", l1 - la)

      -- local arc_scores = scorer:forward(features:t())
      local arc_scores = scorer2:forward({{head_words, head_pos}, {mod_words, mod_pos}})

      l2 = os.clock()
      -- print("features", l2 - l1)

      local loss = parser:forward(arc_scores, target)

      -- test gold
      -- local loss = parser:forward(gold, target)

      for i = 2, parser.argmax:size(1) do
         print(string.format("%d\t%s\t_\t%s\t%s\t_\t%d\t_",
                             i-1, sent[i-1].word,  sent[i-1].tag, sent[i-1].tag, parser.argmax[i]-1))
      end
      print("")
   end
end

function prepare()
   -- Read sentence
   local _, dict =
      read_conll("/home/lingpenk/Data/PTB/PTB_YM/train_final_finepos", 40000)
   -- for i = 1, dict.num_words do
   --    print (dict.index_to_symbol[i])
   -- end
   -- local rfile=io.open("/home/lingpenk/Data/CLPar_Data/multiling_word_embeding/eacl14-data/en_640", "r")
   local rfile=io.open("/home/lingpenk/research/wangling_word2vec/word2vec/wl_vector_readable", "r")

   assert(rfile)
   local word2emd = {}
   for str in rfile:lines() do
      -- embed = split(str, " ")

      -- str = 'cat,dog'
      local w = "UNKNOWN"
      local ind = -1
      local embedding = torch.zeros(word_vec_length):double()
      for word in string.gmatch(str, '([^ ]+)') do
         ind = ind + 1
         if ind == 0 then
            if not dict.symbol_to_index[word] then goto continue end
            w = word
            -- print (w)
         else
            embedding[ind] = tonumber(word)
         end
      end
      word2emd[w] = embedding
      ::continue::
      -- print (embed[0])

   end  
   rfile:close()
   -- print (word2emd["after"])
   return word2emd

end

-- prepare()

main()
 -- test()
