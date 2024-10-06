# isort: off
import sys

sys.path.append("common")
# isort: on
import tree_sitter_utils as ts_utils
import lmpa_ir


class LmpaParser:
    def __init__(self):
        self.all_lmpa_ir = []

    def parse_expression(self, expr):
        if expr.type == 'identifier':
            return lmpa_ir.LmPaVarExpression(var_name=expr.text.decode('utf-8')), True
        elif expr.type == 'call_expression':
            arg_list = ts_utils.get_first_opt(expr, 'argument_list')
            func_id = ts_utils.get_first_opt(expr, 'identifier')
            if func_id is None:
                # all_ids = ts_utils.find_all_recursively(expr, 'identifier')
                # all_ids = [x.text.decode('utf-8') for x in all_ids]
                # lmpa_var_exprs = [lmpa_ir.LmPaVarExpr(var_name=x) for x in all_ids]
                # return lmpa_ir.LmPaBasicExpr(uses=set(lmpa_var_exprs), is_direct_use=False)
                lmpa_expr = lmpa_ir.LmPaExpression()
                lmpa_expr.src_text = expr.text.decode('utf-8')            
                return lmpa_expr, False
            else:
                func_name = func_id.text.decode('utf-8')
                arg_exprs = []
                for arg in arg_list.named_children:
                    arg_expr, _ = self.parse_expression(arg)
                    arg_exprs.append(arg_expr)
                call_expr = lmpa_ir.LmPaCallExpr(func_id=func_name, args=arg_exprs)
                self.all_lmpa_ir.append(call_expr)
                implicit_ret_expr = lmpa_ir.LmPaImplicitReturnVarExpr(var_name=func_name, func_id=func_name)
                return implicit_ret_expr, True
            pass
        elif expr.type == 'assignment_expression':
            non_error_children = [c for c in expr.named_children if c.type != 'ERROR']  
            if len(non_error_children) != 2:
                pass
            else:
                lhs = non_error_children[0]
                rhs = non_error_children[1]
                lhs_expr, lhs_direct = self.parse_expression(lhs)
                rhs_expr, rhs_direct = self.parse_expression(rhs)
                if type(lhs_expr) != lmpa_ir.LmPaVarExpression:
                    dummy_expr = lmpa_ir.LmPaExpression()
                    dummy_expr.src_text = expr.text.decode('utf-8')
                    return dummy_expr, False
                                    
                ret_expr = lmpa_ir.LmPaBasicExpr(defs=[lhs_expr], uses=[rhs_expr], is_direct_use=rhs_direct)
                self.all_lmpa_ir.append(ret_expr)
                return lhs_expr, True
        elif expr.type == 'cast_expression':
            return self.parse_expression(expr.children[-1])
        elif expr.type == 'parenthesized_expression':
            return self.parse_expression(expr.named_children[0])
            
        # all others, simply recursively call
        for child in expr.named_children:
            self.parse_expression(child)
        dummy_expr = lmpa_ir.LmPaExpression()
        dummy_expr.src_text = expr.text.decode('utf-8')
        return dummy_expr, False
    
    def parse_rets(self, rets):
        for ret in rets:
            if len(ret.named_children) >= 1:
                ret_expr = self.parse_expression(ret.named_children[0])
                self.all_lmpa_ir.append(lmpa_ir.LmPaReturnStmt(ret_expr))

    def parse_body(self, root):    
        self.parse_expression(root.root_node)
        rets = ts_utils.find_all_recursively(root.root_node, 'return_statement')
        self.parse_rets(rets)
        
