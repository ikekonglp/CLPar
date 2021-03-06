#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Parser.c"
#else

#ifndef PARSER_H
#define PARSER_H

#define Trap 0
#define Tri  1
#define kSym 2

#define Left 0
#define Right 1
#define kDir 2


typedef struct {
    int sym;
    int dir;
    int s;
    int t;
} Item;

static int item(int length_, int sym_, int dir_, int s_, int t_) {
    return sym_ * kDir * length_ * length_ +
            dir_ * length_ * length_ +
            s_ * length_ +
            t_;
}

Item init_item(int length_, int item_index) {
    int cur = item_index;
    Item item;
    item.sym = cur / (kDir * length_ * length_);
    cur = cur % (kDir * length_ * length_);
    item.dir = cur / (length_ * length_);
    cur = cur % (length_ * length_);
    item.s  = cur / (length_);
    item.t  = cur % (length_);
    return item;
}

typedef struct {
    int item1;
    int item2;
    int terminal;
} BackPointer;

BackPointer *init_backpointer(int item1_, int item2_, int terminal_) {
    BackPointer *bp = malloc(sizeof(*bp));
    bp->item1 = item1_;
    bp->item2 = item2_;
    bp->terminal = terminal_;
    return bp;
}

typedef struct {
    int length_;
    double *scores_;
    BackPointer **bps_;
} Chart;


Chart init_chart(int n) {
    Chart chart;
    chart.length_ = n+1;
    int total_size = chart.length_ * chart.length_ * kSym * kDir;
    chart.scores_ = malloc(total_size*sizeof(*chart.scores_));
    chart.bps_ = malloc(total_size*sizeof(*chart.bps_));
    /* new BackPointer*[total_size]; */
    int i;
    for (i = 0; i < total_size; ++i) {
        chart.bps_[i] = NULL;
    }
    return chart;
}

void free_chart(Chart *chart) {
    int total_size = chart->length_ * chart->length_ * kSym * kDir;
    free(chart->scores_);


    int i;
    for (i = 0; i < total_size; ++i) {
        if (chart->bps_[i] != NULL) {
            free(chart->bps_[i]);
        }
    }
    free(chart->bps_);
    return chart;
}


/* class Chart { */
/*   public: */
/*     Chart(int n) { */
/*         length_ = n+1; */
/*         int total_size = length_ * length_ * kSym * kDir; */
/*         scores_ = new double[total_size]; */
/*         bps_ = new BackPointer*[total_size]; */
/*         int i; */
/*         for (i = 0; i < total_size; ++i) { */
/*             bps_[i] = NULL; */
/*         } */
/*     } */

/* int index(Chart *chart, const Item *item) { */
/*     return item->sym * kDir * chart->length_ * chart->length_ + */
/*             item->dir * chart->length_ * chart->length_ + */
/*             item->s * chart->length_ + */
/*             item->t; */
/* } */

void init(Chart *chart, int item) {
    chart->scores_[item] = 0;
    chart->bps_[item] = init_backpointer(item, item, 1);
}

double score(Chart *chart, int item) {
    return chart->scores_[item];
}

void set(Chart *chart,
         int item,
         int item1,
         int item2,
         double score) {
    /* int item_index1 = index(chart, item1); */
    /* int item_index2 = index(chart, item2); */
    if (chart->bps_[item1] == NULL || chart->bps_[item2] == NULL) {
        return;
    }
    double new_score = chart->scores_[item1] + chart->scores_[item2] + score;

    if (chart->bps_[item] == NULL || new_score > chart->scores_[item]) {
        chart->scores_[item] = new_score;
        chart->bps_[item] = init_backpointer(item1, item2, 0);
    }
}

void finish(Chart *chart,
            int item_index,
            double *deps) {
    BackPointer *bp = chart->bps_[item_index];
    Item item = init_item(chart->length_, item_index);
    /* printf("finish %d %d\n", item.sym, item.dir); */
    if (bp->terminal) return;
    if (item.sym == Trap) {
        if (item.dir == Left) {
            deps[item.s] = item.t;
        } else {
            deps[item.t] = item.s;
        }
    }
    finish(chart, bp->item1, deps);
    finish(chart, bp->item2, deps);
}

  /* private: */

void parse(int n, double *input, double *argmax) {
    Chart chart = init_chart(n);
    int s, k, r;
    for (s = 0; s <= n; ++s) {
        init(&chart, item(n+1, Tri, Right, s, s));
        if (s != 0) {
            init(&chart, item(n+1, Tri, Left, s, s));
        }
    }
    /* printf("chart size: %d\n", n); */
    for (k = 1; k <= n; ++k) {
        for (s = 0; s <= n; s++) {
            int t = k + s;
            if (t > n) break;

            double score1 = input[t * (n+1) + s];
            double score2 = input[s * (n+1) + t];

            for (r = s; r < t; ++r) {
                if (s != 0) {
                    set(&chart,
                        item(n+1, Trap, Left, s, t),
                        item(n+1, Tri, Right, s, r),
                        item(n+1, Tri, Left, r+1, t),
                        score1);
                }
                set(&chart,
                    item(n+1, Trap, Right, s, t),
                    item(n+1, Tri, Right, s, r),
                    item(n+1, Tri, Left, r+1, t),
                    score2);
            }

            for (r = s; r < t; ++r) {
                if (s != 0) {
                    set(&chart,
                        item(n+1, Tri, Left, s, t),
                        item(n+1, Tri, Left, s, r),
                        item(n+1, Trap, Left, r, t),
                        0);
                }
                set(&chart,
                    item(n+1, Tri, Right, s, t),
                    item(n+1, Trap, Right, s, r+1),
                    item(n+1, Tri, Right, r+1, t),
                    0);
            }
        }
    }
    double scor = score(&chart, item(n+1, Tri, Right, 0, n));
    /* printf("finishing %f\n", scor); */
    finish(&chart, item(n+1, Tri, Right, 0, n), argmax);
    free_chart(&chart);
}

#endif /* PARSER_H */

static int parse_(Parser_updateOutput)(lua_State *L)
{
    THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *target = luaT_checkudata(L, 3, torch_Tensor);
    THTensor *argmax = luaT_getfieldcheckudata(L, 1, "argmax", torch_Tensor);

    int size = THTensor_(size)(target, 0);

    /* printf("Size %d\n", size); */
    THTensor_(resizeAs)(argmax, target);
    parse(size-1, THTensor_(data)(input), THTensor_(data)(argmax));
    return 1;
}

static int parse_(Parser_updateGradInput)(lua_State *L)
{
    THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
    THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
    THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

    THTensor_(resizeAs)(gradInput, input);
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input, \
                     real z = *input_data;                              \
                     *gradInput_data = *gradOutput_data * (z >= 0 ? 1 : -1);)
            return 1;
}

static const struct luaL_Reg parse_(Parser__) [] = {
    {"Parser_updateOutput", parse_(Parser_updateOutput)},
    {"Parser_updateGradInput", parse_(Parser_updateGradInput)},
    {NULL, NULL}
};

static void parse_(Parser_init)(lua_State *L)
{
    luaT_pushmetatable(L, torch_Tensor);
    luaT_registeratname(L, parse_(Parser__), "parse");
    lua_pop(L,1);
}


#endif
