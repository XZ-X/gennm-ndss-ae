class LmPaExpression:
    def __init__(self):
        self.src_text = ''


class LmPaVarExpression(LmPaExpression):
    def __init__(self, var_name):
        super().__init__()
        self.var_name = var_name
        self.defs = set()
        self.uses = set()

    def __str__(self):
        ret = "Var: %s" % self.var_name
        return ret

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, LmPaVarExpression):
            return self.var_name == other.var_name
        else:
            return False


class LmPaImplicitReturnVarExpr(LmPaExpression):
    def __init__(self, var_name, func_id):
        super().__init__()
        self.var_name = var_name
        self.func_id = func_id

    def __str__(self):
        ret = "ImplicitReturnVar: %s(%s)" % (self.var_name, self.func_id)
        return ret

    def __repr__(self):
        return self.__str__()


class LmPaBasicExpr(LmPaExpression):
    def __init__(self, defs, uses, is_direct_use):
        super().__init__()
        self.defs = defs
        self.uses = uses
        self.is_direct_use = is_direct_use

    def __str__(self):
        ret = "BasicExpr: "
        ret += "defs: %s, " % str(self.defs)
        ret += "uses: %s, " % str(self.uses)
        return ret

    def __repr__(self):
        return self.__str__()


class LmPaCallExpr(LmPaExpression):
    def __init__(self, func_id, args):
        super().__init__()
        self.func_id = func_id
        self.args = args

    def __str__(self):
        ret = "CallExpr: %s(%s)" % (self.func_id, str(self.args))
        return ret

    def __repr__(self):
        return self.__str__()


class LmPaReturnStmt(LmPaExpression):
    def __init__(self, ret_val):
        super().__init__()
        self.ret_val = ret_val

    def __str__(self):
        ret = "ReturnStmt: %s" % str(self.ret_val)
        return ret

    def __repr__(self):
        return self.__str__()


class LmPaBranchStmt(LmPaExpression):
    def __init__(self, successors):
        super().__init__()
        self.successors = successors

    def __str__(self):
        ret = "BranchStmt: %s" % str(self.successors)
        return ret

    def __repr__(self):
        return self.__str__()


class LmPaBasicBlock:
    def __init__(self, block_id, exprs, is_terminated):
        self.block_id = block_id
        self.exprs = exprs
        self.is_terminated = is_terminated
        self.successors = []
        self.predecessors = []

    def __str__(self):
        ret = "BB: %s" % (self.block_id)
        return ret

    def __repr__(self):
        return self.__str__()


class LmPaFunction:
    def __init__(self, func_id, args, blocks):
        self.func_id = func_id
        self.args = args
        self.blocks = blocks

    def __str__(self):
        ret = "Func: %s" % (self.func_id)
        return ret

    def __repr__(self):
        return self.__str__()
