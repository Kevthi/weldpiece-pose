

def nice_print_dict(a_dict, indent=0):
    for key in a_dict:
        val = a_dict[key]
        if type(val) is dict:
            print(indent*"   " + key)
            nice_print_dict(val, indent+1)
        else:
            print(indent*"   ", key,val)
