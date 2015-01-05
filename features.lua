embeddingDim = 50
featureLimit = 100000000

-- Feature states.
offset = {}
function offset.make(sizes)

   local offset = {}
   offset.sizes = sizes

   offset.total_size = 1
   for i = 1, #sizes do
      offset.total_size = offset.total_size * sizes[i]
   end
   return offset
end

feature_state = {}
function feature_state.make(offsets, base, size)
   local state = {}
   state.base = base
   state.offset = offsets
   state.feature_num = 1
   state.tally = 1 -- torch.zeros(size):long()
   state.tmp = torch.zeros(size):long()
   return state
end

function feature_state.inc(state, feature)
   local t = state.offset[state.feature_num]
   local spot = state.base[state.feature_num]
   local multiplier = 1
   assert(#feature == #t.sizes)
   for i = 1, #feature do
      torch.add(spot, spot, multiplier, feature[i])
      multiplier = multiplier * t.sizes[i]
   end
   torch.add(spot, spot, state.tally)
   state.tally = state.tally + t.total_size;
   state.feature_num = state.feature_num + 1
end

-- Templates (a subset of the arc-features)
function feature_templates(dict)
   local offsets = {}

   for i = 1, 2 do
      local dir = 1
      local distance = 1
      if i == 2 then
         dir = 2
         distance = 10
      end

      local templates = {{dict.num_words, dir, distance},
                         {dict.num_words, dir, distance},
                         {dict.num_tags, dir, distance},
                         {dict.num_tags, dir, distance},
                         {dict.num_words, dict.num_tags, dir, distance},
                         {dict.num_words, dict.num_tags, dir, distance},
                         {dict.num_words, dict.num_words, dir, distance},
                         {dict.num_words, dict.num_tags, dir, distance},
                         {dict.num_tags, dict.num_words, dir, distance},
                         {dict.num_tags, dict.num_tags, dir, distance},
                         {dict.num_words, dict.num_tags, dict.num_tags, dir, distance},
                         {dict.num_words, dict.num_tags, dict.num_tags, dir, distance},
                         {dict.num_tags, dict.num_tags, dict.num_tags, dict.num_tags, dir, distance},
                         {dict.num_tags, dict.num_tags, dict.num_tags, dict.num_tags, dir, distance},
                         {dict.num_tags, dict.num_tags, dict.num_tags, dict.num_tags, dir, distance},
                         {dict.num_tags, dict.num_tags, dict.num_tags, dict.num_tags, dir, distance}}

      for i = 1, #templates do
         table.insert(offsets, offset.make(templates[i]))
      end
   end
   return offsets
end

function bucket(dist)
   if dist == 1 then
      return 1
   elseif dist == 2 then
      return 2
   elseif dist == 3 then
      return 3
   elseif dist == 4 then
      return 4
   elseif dist < 8 then
      return 5
   elseif dist < 15 then
      return 6
   elseif dist < 20 then
      return 7
   elseif dist < 40 then
      return 8
   else
      return 9
   end
end

function features_mat(sent, parts, offsets, features)
   local n = sent:size(1)

   local WORD = sent[1]
   local POS = sent[2]
   local HEAD = parts:t()[1]
   HEAD:long()
   local MOD = parts:t()[2]
   MOD:long()

   local head_words = WORD:index(1, HEAD)
   local mod_words = WORD:index(1, MOD)

   local head_tags = POS:index(1, HEAD)
   local mod_tags = POS:index(1, MOD)


   local modl = (MOD-1)
   modl[modl:le(1)] = 1
   local modr = (MOD+1)
   modr[modr:ge(n)] = n
   local headl = (HEAD-1)
   headl[headl:le(1)] = 1
   local headr = (HEAD+1)
   headr[headr:ge(n)] = n

   local head_l_tags = POS:index(1, headl)
   local head_r_tags = POS:index(1, headr)
   local mod_l_tags = POS:index(1, modl)
   local mod_r_tags = POS:index(1, modr)

   local dir = HEAD:ge(MOD):long()
   local distance = (HEAD - MOD):apply(bucket):long()
   local e = torch.ones(parts:size(1)):long()
   -- local features = torch.ones(#offsets, parts:size(1)):long()

   features:zero()
   s = feature_state.make(offsets, features, parts:size(1))
   for i = 1, 2 do
      local ex1 = e
      local ex2 = e
      if i == 2 then
         ex1 = dir
         ex2 = distance
      end
      local features = {{head_words, ex1, ex2},
                        {mod_words, ex1, ex2},
                        {head_tags, ex1, ex2},
                        {mod_tags, ex1, ex2},
                        {head_words, head_tags, ex1, ex2},
                        {mod_words, mod_tags, ex1, ex2},
                        {head_words, mod_words, ex1, ex2},
                        {head_words, mod_tags, ex1, ex2},
                        {head_tags, mod_words, ex1, ex2},
                        {head_tags, mod_tags, ex1, ex2},
                        {head_words, mod_tags, head_tags, ex1, ex2},
                        {mod_words, mod_tags, head_tags, ex1, ex2},
                        {mod_l_tags, mod_tags, head_l_tags, head_tags, ex1, ex2},
                        {mod_r_tags, mod_tags, head_l_tags, head_tags, ex1, ex2},
                        {mod_l_tags, mod_tags, head_r_tags, head_tags, ex1, ex2},
                        {mod_r_tags, mod_tags, head_r_tags, head_tags, ex1, ex2}}


      for j, feature in pairs(features) do
         feature_state.inc(s, feature)
      end
   end
   -- features:clamp(featureLimit)
   -- print(features)
   features:apply(function(a) return math.abs(a) % (featureLimit-1) + 1; end)

end


-- Features
function features(sent, parts, offsets)
   local WORD = 1
   local POS = 2
   local HEAD = 1
   local MOD = 2

   local features = torch.zeros(parts:size(1), #offsets):long()
   -- print(parts)
   for i = 1, parts:size(1) do
      s = feature_state.make(offsets, features[i])
      feature_state.inc(s, sent[WORD][parts[i][HEAD]], 1, 1, 1)
      feature_state.inc(s, sent[WORD][parts[i][MOD]], 1, 1, 1)
      feature_state.inc(s, sent[POS][parts[i][HEAD]], 1, 1, 1)
      feature_state.inc(s, sent[POS][parts[i][MOD]], 1, 1, 1)
      feature_state.inc(s, sent[WORD][parts[i][HEAD]], sent[POS][parts[i][HEAD]], 1, 1)
      feature_state.inc(s, sent[WORD][parts[i][MOD]], sent[POS][parts[i][MOD]], 1, 1)

      feature_state.inc(s, sent[WORD][parts[i][HEAD]], sent[WORD][parts[i][MOD]], 1, 1)
      feature_state.inc(s, sent[WORD][parts[i][HEAD]], sent[POS][parts[i][MOD]], 1, 1)
      feature_state.inc(s, sent[POS][parts[i][HEAD]], sent[WORD][parts[i][MOD]], 1, 1)
      feature_state.inc(s, sent[POS][parts[i][HEAD]], sent[POS][parts[i][MOD]], 1, 1)
      feature_state.inc(s, sent[WORD][parts[i][HEAD]], sent[POS][parts[i][HEAD]],
                        sent[POS][parts[i][MOD]], 1)
      feature_state.inc(s, sent[WORD][parts[i][MOD]], sent[POS][parts[i][HEAD]],
                        sent[POS][parts[i][MOD]], 1)
   end
   return features
end
