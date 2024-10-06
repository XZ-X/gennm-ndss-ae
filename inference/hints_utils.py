


def mix_hints_w_code(body, hints, max_callee=5, max_callsite=5):
    hints_string = ""
    if len(hints['callees']) != 0:
        hints_string += "<Hints Callees>"        
        seen_callees = set()
        for callee_name, callee_hints in hints['callees']:
            callee_hints = callee_hints.strip()
            if callee_hints in seen_callees:
                continue
            seen_callees.add(callee_hints)
            if len(seen_callees) >= max_callee:
                break            
            hints_string += "\n{callee_name}: {callee_hints}".format(
                callee_name=callee_name, callee_hints=callee_hints
            )
        hints_string += "\n</Hints Callees>\n"
    if len(hints['callsites']) != 0:
        seen_callsites = set()
        hints_string += "<Hints Callsites>"
        for callsite_hints in hints['callsites']:            
            callsite_hints = callsite_hints.strip()
            if callsite_hints in seen_callsites:
                continue
            if len(seen_callsites) >= max_callsite:
                break
            id = len(seen_callsites)
            seen_callsites.add(callsite_hints)
            hints_string += "\n<{id}> {callsite_hints} </{id}>".format(
                callsite_hints=callsite_hints, id=id
            )
        hints_string += "\n</Hints Callsites>"
    if hints_string == "":
        return body, ""
    return body + "\n" + hints_string, hints_string


