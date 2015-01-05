

function test_parse()
   local n = 4

   local parser = nn.Parser()
   local input = torch.ones(n * n)
   for test = 1, 4 do
      -- print("START")
      local parse = first_parse(n)
      for i = 1, n*n do
         input[i] = torch.uniform()
      end
      local score = 0
      local best_parse = torch.Tensor(n)
      while true do
         parse = next_parse(parse)
         if parse == nil then
            break
         end

         local tmp_score = parser:score(input, parse)
         if tmp_score > score then
            best_parse:copy(parse)
            score = tmp_score
         end
      end

      local correct = parser:forward(input, best_parse)
      -- print(best_parse, score)
      -- print(parser.argmax, parser:score(input, parser.argmax))
      -- print(correct)
   end

end
