
import name_utils
import re
import networkx as nx


class Function:
    def __init__(self, func_name, body, var_id_maps, func_id_maps):
        """
        Args:
            body: str, stripped function body
            var_id_maps: dict, key is current var name, value is ground truth var name
            func_id_maps: dict, key is current func name, value is ground truth func name
        """
        self.body = body
        self.var_id_maps = var_id_maps
        self.func_id_maps = func_id_maps
        self.func_name = func_name
        self.rename_map = {}
    


class BinaryProgram:
    KEY_ADDR = 'e'
    KEY_GT_FUNC = 'b'
    KEY_STRIPPED_FUNC = 'c'
    KEY_NAME = 'n'
    KEY_CODE = 'c'
    KEY_ARGS = 'a'
    KEY_LOCALs = 'l'
    KEY_VAR_NAME = 'n'
    FUNC_PATTERN = re.compile(r'([^a-zA-Z0-9_]|^)(sub_[0-9a-fA-F_]*)([^a-zA-Z0-9_])')

    def __init__(self, prog_name, func_entry_list):
        self.prog_name = prog_name
        self.funcs_all = func_entry_list
        self.funcs_filtered = [func for func in func_entry_list if name_utils.is_interesting_func(func)]
        self.stripped_name2entry = {
            func[BinaryProgram.KEY_STRIPPED_FUNC][BinaryProgram.KEY_NAME]:func for func in self.funcs_filtered
        }
        self.call_graph = nx.DiGraph()
        # add all func to nodes
        for func_name in self.stripped_name2entry.keys():
            if func_name is None:
                continue
            self.call_graph.add_node(func_name)
        self._gen_call_graph()
        self.stripped_name2func = {}
        for func_entry in self.funcs_filtered:
            func_name = func_entry[BinaryProgram.KEY_STRIPPED_FUNC][BinaryProgram.KEY_NAME]
            var_id_maps = self._collect_vars(func_entry)
            func_id_maps = self._collect_funcs(func_entry)
            func_body = func_entry[BinaryProgram.KEY_STRIPPED_FUNC][BinaryProgram.KEY_CODE]
            if len(var_id_maps) + len(func_id_maps) > 0:
                self.stripped_name2func[func_name] = Function(func_name, func_body, var_id_maps, func_id_maps)
                
    
    def _gen_call_graph(self):
        for func in self.funcs_filtered:
            stripped_func = func[BinaryProgram.KEY_STRIPPED_FUNC]
            stripped_func_name = stripped_func[BinaryProgram.KEY_NAME]
            stripped_func_code = stripped_func[BinaryProgram.KEY_CODE]
            for match in BinaryProgram.FUNC_PATTERN.finditer(stripped_func_code):
                callee_name = match.group(2)
                if callee_name == stripped_func_name:
                    continue
                if callee_name in self.stripped_name2entry:
                    if stripped_func_name is None or callee_name is None:
                        continue
                    self.call_graph.add_edge(stripped_func_name, callee_name)


    def _collect_vars(self, func_entry):
        var_id_maps = {}
        gt_pos2var = {}
        gt_func = func_entry[BinaryProgram.KEY_GT_FUNC]
        stripped_func = func_entry[BinaryProgram.KEY_STRIPPED_FUNC]
        for arg_pos, var_entry in gt_func[BinaryProgram.KEY_ARGS].items():
            gt_pos2var[arg_pos] = var_entry[0][BinaryProgram.KEY_VAR_NAME]
        
        for arg_pos, var_entry in stripped_func[BinaryProgram.KEY_ARGS].items():
            if arg_pos in gt_pos2var:
                stripped_name = var_entry[0][BinaryProgram.KEY_VAR_NAME]
                gt_name = gt_pos2var[arg_pos]
                if name_utils.is_interesting_name(gt_name) and gt_name != stripped_name:
                    var_id_maps[stripped_name] = gt_name                
        
        for local_pos, var_entry in gt_func[BinaryProgram.KEY_LOCALs].items():
            gt_pos2var[local_pos] = var_entry[0][BinaryProgram.KEY_VAR_NAME]
        for local_pos, var_entry in stripped_func[BinaryProgram.KEY_LOCALs].items():
            if local_pos in gt_pos2var:
                stripped_name = var_entry[0][BinaryProgram.KEY_VAR_NAME]
                gt_name = gt_pos2var[local_pos]
                if name_utils.is_interesting_name(gt_name) and gt_name != stripped_name:
                    var_id_maps[stripped_name] = gt_name

        return var_id_maps
    
    def _collect_funcs(self, func_entry):
        func_id_maps = {}
        # function ids
        for match in BinaryProgram.FUNC_PATTERN.finditer(func_entry[BinaryProgram.KEY_STRIPPED_FUNC][BinaryProgram.KEY_CODE]):
            callee_name = match.group(2)            
            if callee_name in self.stripped_name2entry:
                gt_name = self.stripped_name2entry[callee_name][BinaryProgram.KEY_GT_FUNC][BinaryProgram.KEY_NAME]
                if gt_name != callee_name:
                    func_id_maps[callee_name] = gt_name
        return func_id_maps
