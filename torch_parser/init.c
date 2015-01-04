#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define parse_(NAME) TH_CONCAT_3(parse_, Real, NAME)

#include "generic/Parser.c"
#include "THGenerateFloatTypes.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libparse(lua_State *L);

int luaopen_libparse(lua_State *L)
{
    lua_newtable(L);
    lua_pushvalue(L, -1);
    lua_setfield(L, LUA_GLOBALSINDEX, "parse");

    parse_FloatParser_init(L);
    parse_DoubleParser_init(L);


    return 1;
}
