embeddingDim = 50
featureLimit = 1000000

-- Feature states.
offset = {}
function offset.make(size_a, size_b, size_c, size_d)
   local offset = {}
   offset.size_a = size_a
   offset.size_b = size_b
   offset.size_c = size_c
   offset.size_d = size_d
   offset.total_size = size_a * size_b * size_c * size_d;
   return offset
end

feature_state = {}
function feature_state.make(offsets, base)
   local state = {}
   state.base = base
   state.offset = offsets
   state.feature_num = 1
   state.tally = 0
   return state
end

function feature_state.inc(state, a, b, c, d)
   local t = state.offset[state.feature_num]
   local app = (a * t.size_b * t.size_c * t.size_d +
                   b * t.size_c * t.size_c * t.size_d +
                   c * t.size_d
                   + d);

   local index = state.tally + app;
   state.base[state.feature_num] = index % featureLimit
   -- table.insert(state.base, index)

   state.tally = state.tally + t.total_size;
   state.feature_num = state.feature_num + 1;
end

-- Templates
function feature_templates(dict)
   local offsets = {}

   local templates = {{dict.num_words, 1, 1, 1},
    {dict.num_words, 1, 1, 1},
    {dict.num_tags, 1, 1, 1},
    {dict.num_tags, 1, 1, 1},
    {dict.num_words, dict.num_tags, 1, 1},
    {dict.num_words, dict.num_tags, 1, 1},
    {dict.num_words, dict.num_words, 1, 1},
    {dict.num_words, dict.num_tags, 1, 1},
    {dict.num_tags, dict.num_words, 1, 1},
    {dict.num_tags, dict.num_tags, 1, 1},
    {dict.num_words, dict.num_tags, dict.num_tags, 1},
    {dict.num_words, dict.num_tags, dict.num_tags, 1}}

   for i = 1, #templates do
      table.insert(offsets, offset.make(templates[i][1],
                                        templates[i][2],
                                        templates[i][3],
                                        templates[i][4]))
   end
   return offsets
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
