import copy


def get_initializers_info(model):
    init_usage_map = {init.name: [] for init in model.graph.initializer}
    for node in model.graph.node:
        for input_name in node.input:
            if input_name in init_usage_map:
                init_usage_map[input_name].append(node)

    name_2_init = {init.name: init for init in model.graph.initializer}

    return init_usage_map, name_2_init


def duplicate_shared_initializers(model):
    init_usage_map, name_2_init = get_initializers_info(model)

    for init_name, nodes in init_usage_map.items():

        if len(nodes) <= 1:
            continue
        print(
            f"Initializer '{init_name}' is reused {len(nodes)} times, apply deepcopy...")

        for idx, node in enumerate(nodes):
            flag = False
            idx_del = 0
            for idx_re, i_name in enumerate(node.input):
                if i_name == init_name:
                    print(f"node:{node.name} initializer:{init_name}")
                    flag = True
                    idx_del = idx_re
                    break
            get_init = name_2_init.get(init_name, None)
            if get_init is None:
                continue

            new_init = copy.deepcopy(get_init)
            print(f"initializer of node:{node.name} {init_name} copy complete")
            new_name = f"{init_name}_copy{idx}"
            new_init.name = new_name
            model.graph.initializer.append(new_init)
            node.input[idx_del] = new_name
