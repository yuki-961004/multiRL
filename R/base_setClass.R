################################## [input] #####################################

methods::setClass(
  Class = "multiRL.colnames",
  slots = list(
    subid = "character", 
    block = "character",
    trial = "character",
    object = "character", 
    reward = "character",
    action = "character",
    exinfo = "character"
  )
)

methods::setClass(
  Class = "multiRL.params",
  slots = list(
    free = "list",
    fixed = "list", 
    constant = "list"
  )
)

methods::setClass(
  Class = "multiRL.funcs",
  slots = list(
    lrng_func = "function", 
    prob_func = "function",
    util_func = "function",
    bias_func = "function",
    expl_func = "function",
    dcay_func = "function"
  )
)

methods::setClass(
  Class = "multiRL.features",
  slots = list(
    idinfo = "array",
    state = "array",
    action = "array",
    exinfo = "array"
  )
)

methods::setClass(
  Class = "multiRL.settings",
  slots = list(
    name = "character",
    mode = "character",
    estimate = "character",
    policy = "character",
    system = "character"
  )
)

methods::setClassUnion("numericORcharacter", c("numeric", "character"))

methods::setClass(
  Class = "multiRL.input",
  slots = list(
    data = "data.frame",
    colnames = "multiRL.colnames",
    features = "multiRL.features",
    params = "multiRL.params",
    priors = "list",
    funcs = "multiRL.funcs",
    settings = "multiRL.settings",
    elements = "numeric",
    subid = "character",
    n_block = "numeric",
    n_trial = "numericORcharacter",
    n_rows = "numeric",
    extra = "list"
  )
)

################################# [behrule] ####################################

methods::setClass(
  Class = "multiRL.behrule",
  slots = list(
    cue = "character", 
    mid = "character",
    rsp = "character",
    extra = "list"
  )
)

################################# [record] #####################################

methods::setClass(
  Class = "multiRL.result",
  slots = list(
    value = "list",
    bias = "matrix",
    shown = "matrix",
    prob = "matrix",
    count = "matrix",

    hidden = "matrix",

    exploration = "matrix",
    latent = "matrix",
    reward = "matrix",
    utility = "matrix",
    simulation = "matrix",
    position = "matrix"
  )
)

methods::setClass(
  Class = "multiRL.record",
  slots = list(
    input = "multiRL.input",
    behrule = "multiRL.behrule",
    result = "multiRL.result",
    extra = "list"
  )
)

################################# [output] #####################################

methods::setClass(
  Class = "multiRL.output",
  slots = methods::getSlots("multiRL.record"),
  contains = "multiRL.record"
)

################################# [metric] #####################################

methods::setClass(
  Class = "multiRL.sumstat",
  slots = list(
    ACC = "numeric",
    LL = "numeric",
    AIC = "numeric",
    BIC = "numeric",
    LPr = "numeric",
    LPo = "numeric",
    ABC = "list",
    extra = "list"
  )
)

methods::setClass(
  Class = "multiRL.metric",
  slots = list(
    input = "multiRL.input",
    behrule = "multiRL.behrule",
    result = "multiRL.result",
    sumstat = "multiRL.sumstat",
    extra = "list"
  )
)

################################## [model] #####################################

methods::setClass(
  Class = "multiRL.model",
  slots = methods::getSlots("multiRL.metric"),
  contains = "multiRL.metric"
)

methods::setClass(
  Class = "multiRL.summary",
  slots = list(
    data = "data.frame",
    process = "list",
    params = "multiRL.params",
    metrics = "multiRL.sumstat"
  )
)
