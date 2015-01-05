function string:split(sep)
   local sep, fields = sep or ":", {}
   local pattern = string.format("([^%s]+)", sep)
   self:gsub(pattern, function(c) fields[#fields+1] = c end)
        return fields
end

function generate_parts(length)
   local parts = torch.ones(length*length, 2):long()
   local row = 1
   for i = 1, length do
      for j = 1, length do
         -- if i ~= j then
            parts[row][1] = i
            parts[row][2] = j
            row = row + 1
         -- end
      end
   end
   return parts
end

-- A dependency parse is a torch.Tensor(n+1)
-- Word 1 is the root.
function is_projective(parse)
   for m = 1, parse:size(1) do
      local h = parse[m]

      for m2 = 1, parse:size(1) do
         local h2 = parse[m2]
         if m2 ~= m then
            if m < h then
               if (m < m2 and m2 < h and h < h2) or
                  (m < h2 and h2 < h and h < m2) or
                  (m2 < m and m < h2 and h2 < h) or
                  (h2 < m  and m < m2 and m2 < h) then
                  return false
               end
            end
            if h < m then
               if (h < m2 and m2 < m and m < h2) or
                  (h < h2 and h2 < m and m < m2) or
                  (m2 < h and h < h2 and h2 < m) or
                  (h2 < h and h < m2 and m2 < m) then
                  return false
               end
            end
         end
      end
   end
   return true
end

function is_spanning(parse)
   local children = {}
   for m = 2, parse:size(1) do
      local h = parse[m]
      if m == h then
         return false
      end
      children[h] = children[h] or {}
      table.insert(children[h], m)
   end
   local stack = {1}
   local seen = {}
   while #stack > 0 do
      cur = stack[#stack]
      if seen[cur] then
         return false
      end
      seen[cur] = true
      table.remove(stack)
      if children[cur] then
         for i = 1, #children[cur] do
            table.insert(stack, children[cur][i])
         end
      end
   end
   if #seen ~= parse:size(1)  then
      return false
   end
   return true
end

function next_product(vector)
   local n = vector:size(1)
   for i = 1, vector:size(1) do
      vector[i] = vector[i] + 1
      if vector[i] > n then
         vector[i] = 1
      else
         return true
      end
   end
   return false
end

function first_parse(n)
   local parse = torch.ones(n):long()
   return parse
end

function next_parse(parse)
   while true do
      if not next_product(parse) then
         break
      end

      if is_projective(parse) and is_spanning(parse) then
         return parse
      end
   end
   return nil
end


function read_conll(fname, limit)
   local f = io.open(fname)
   local sentences = {}
   local dict = {}
   dict.index_to_symbol = {}
   dict.symbol_to_index = {}
   dict.index_to_tag = {}
   dict.tag_to_index = {}

   dict.num_words = 0
   dict.num_tags = 0
   sent = {}
   for line in f:lines() do
      if limit ~= nil and #sentences > limit then
         break
      end
      if line == '' then
         table.insert(sentences, sent)
         sent = {}
      else
         local t = line:split("\t")
         local word = t[2]
         local tag = t[4]
         local head = tonumber(t[7])
         table.insert(sent, {["word"] = word,
                             ["tag"] = tag,
                             ["head"] = head})

         -- Register the word.
         if dict.symbol_to_index[word] == nil then
            dict.num_words = dict.num_words + 1
            dict.symbol_to_index[word] = dict.num_words
            dict.index_to_symbol[dict.num_words] = word
         end

         -- Register the tag.
         if dict.tag_to_index[tag] == nil then
            dict.num_tags = dict.num_tags + 1
            dict.tag_to_index[tag] = dict.num_tags
            dict.index_to_symbol[dict.num_tags] = tag
         end
      end
   end
   return sentences, dict
end
