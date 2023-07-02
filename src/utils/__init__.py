import re
import numpy as np


def get_ordered_objects(object_names, line):
    objs = []
    pos = []
    for obj in object_names:
        if obj in line:
            objs.append(obj)
            pos.append(line.index(obj))

    sorted_zipped_lists = sorted(zip(pos, objs))
    return [el for _, el in sorted_zipped_lists]


def generate_all_actions(state):
    return_list = []
    if "hand is empty" in state:
        block = re.findall("the [a-z]{0,10} block is clear", state)
        block_color = [re.search("the ([a-z]{0,10}) block is clear", b).group(1) for b in block]
        # print(block_color)
        for c in block_color:
            # print("looking for", c)
            if f"the {c} block is on the table" in state:
                return_list.append(f"Pick up the {c} block")
            else:
                c_ = re.search(f"the {c} block" + " is on top of the ([a-z]{0,10}) block", state).group(1)
                return_list.append(f"Unstack the {c} block from on top of the {c_} block")
    else:
        c = re.search("is holding the ([a-z]{0,10}) block", state).group(1)
        block = re.findall("the [a-z]{0,10} block is clear", state)
        clear_color = [re.search("the ([a-z]{0,10}) block is clear", b).group(1) for b in block]
        for c_ in clear_color:
            return_list.append(f"Stack the {c} block on top of the {c_} block")
        return_list.append(f"Put down the {c} block")
    return return_list


def apply_change(change, state):
    # print("input state:", state)
    if "and the " in state and ", and the" not in state:
        state = state.replace("and the ", ", and the ")
    states = state.split(", ")
    states = [s.strip()[4:].strip(".") if s.strip().startswith("and ") else s.strip().strip(".") for s in states]
    # print("state", states)

    changes = change.lower().strip().strip(".").split(", ")
    # print("initial states:", states)
    for c in changes:
        if c.startswith("and "):
            c = c[4:]
        success = 0
        # print("current change", c)
        if c.startswith("the hand"):
            # print(c)
            old = c.split("was")[1].split("and")[0].strip()
            # print(old)
            new = c.split("now")[1].strip()
            # print(new)
            for idx in range(len(states)):
                # print("=", s)
                if ("hand is " + old) in states[idx]:
                    # print(":", s)
                    states[idx] = states[idx].replace(old, new)
                    success += 1
                    # print(s)
        else:
            
            colors = re.findall(r"the (\w+) block", c)
            if len(colors) == 0:
                print("Error: zero-colors")
                print(c)
                raise Exception("ERROR")
            color = colors[0]
            # print(colors)
            if c.startswith(f"the {color} block"):
                # print("SUB:", f"the {color} block")
                subj = f"{color} block"
                if "no longer" in c:
                    old = c.split("no longer")[1].strip()
                    # print("old:", old)
                    for idx in range(len(states)):
                        if f"{color} block is " + old in states[idx]:
                            states[idx] = ""
                            success += 1
                elif "was" in c and "now" in c:
                    old = c.split("was")[1].split(" and")[0].strip()
                    new = c.split("now")[1].strip()
                    # print("previous:", "{color} block is " + old)
                    for idx in range(len(states)):
                        if f"{color} block is " + old in states[idx]:
                            states[idx] = states[idx].replace(old, new)
                            success += 1
                elif "now" in c:
                    new = c.split("now")[1].strip()
                    states.append("the " + color + " block is " + new)
                    success += 1
            else:
                # print("ERROR")
                print("Error: not recognized")
                raise Exception("ERROR")
        if success == 0:
            # print("ERROR")
            print("Error: no successful change")
            print(c)
            print(states)
            raise Exception("ERROR")
        # print("current states:", states)
    states = [s for s in states if s != ""]
    priority_states = []
    for s in states:
        if "have that" in s:
            priority_states.append(0)
        elif "clear" in s:
            priority_states.append(1)
        elif "in the hand" in s:
            priority_states.append(1)
        elif "the hand is" in s:
            priority_states.append(2)
        elif "on top of" in s:
            priority_states.append(3)
        elif "on the table" in s:
            priority_states.append(4)
        else:
            print("Error: unknown state")
            print(s)
            torch.distributed.barrier()
            raise Exception("ERROR")
    sorted_states = [x.strip() for _, x in sorted(zip(priority_states, states))]
    sorted_states[-1] = "and " + sorted_states[-1]
    return ", ".join(sorted_states) + "."


def parsed_instance_to_text_blocksworld(initial_state, plan, goal_state, data, action_seq=False):
    DATA = data
    INIT = ""
    init_text = []
    for i in sorted(initial_state):
        pred = i.split('_')
        objs = [DATA["encoded_objects"][j] for j in pred[1:]]
        init_text.append(DATA['predicates'][pred[0]].format(*objs))
    init_text = init_text
    if len(init_text) > 1:
        INIT += ", ".join(init_text[:-1]) + f" and {init_text[-1]}"
    else:
        INIT += init_text[0]
    INIT += "."
    PLAN = ""
    plan_text = "\n"
    for i in plan:
        pred = i.split('_')
        objs = [DATA["encoded_objects"][j] for j in pred[1:]]
        plan_text += DATA['actions'][pred[0]].format(*objs)
        plan_text += "\n"
    if not action_seq:
        plan_text += "[PLAN END]\n"
    else:
        plan_text += "[ACTION SEQUENCE END]\n"
    PLAN += plan_text

    GOAL = ""
    goal_text = []
    for i in sorted(goal_state):
        pred = i.split('_')
        objs = [DATA["encoded_objects"][j] for j in pred[1:]]
        goal_text.append(DATA['predicates'][pred[0]].format(*objs))
    goal_text = goal_text
    if len(goal_text) > 1:
        GOAL += ", ".join(goal_text[:-1]) + f" and {goal_text[-1]}"
    elif len(goal_text) == 1:
        GOAL += goal_text[0]

    return INIT, PLAN, GOAL


def fill_template(INIT, GOAL, PLAN, data, instruction=False):
    """From https://github.com/karthikv792/LLMs-Planning/blob/9bd247838ec314f048f6a37a147201e74d64572d/plan-bench/utils/pddl_to_text.py#L59"""
    text = ""
    if INIT != "":
        text += "\n[STATEMENT]\n"
        text += f"As initial conditions I have that, {INIT.strip()}."
    if GOAL != "":
        text += f"\nMy goal is to have that {GOAL}."
    if not instruction:
        text += f"\n\nMy plan is as follows:\n\n[PLAN]{PLAN}"
    else:
        text += f"\n\nWhat is the plan to achieve my goal? Just give the actions in the plan."

    if 'blocksworld' in data['domain_name']:
        text = text.replace("-", " ").replace("ontable", "on the table")
    return text


def text_to_plan_blocksworld(text, action_set, plan_file, data, ground_flag=False):
    """
    Converts blocksworld plan in plain text to PDDL plan
    ASSUMPTIONS:
        (1) Actions in the text we have them on the domain file
        (2) We know the object names
        (3) Objects order is given by the sentence

    :param text: Blocksworld text to convert
    :param action_set: Set of possible actions
    :param plan_file: File to store PDDL plan
    """

    # ----------- GET DICTIONARIES ----------- #
    LD = data['encoded_objects']  # Letters Dictionary
    BD = {v: k for k, v in LD.items()}  # Blocks Dictionary
    AD = {}  # Action Dictionary
    for k, v in data['actions'].items():
        word = v.split(' ')[0]
        if word in k:
            AD[k] = k.replace("-", " ")
        else:
            AD[k] = word

    # ----------- GET RAW AND TEXT-FORMATTED ACTIONS AND OBJECTS ----------- #
    actions_params_dict = dict(action_set.items())
    raw_actions = [i.lower() for i in list(action_set.keys())]
    text_actions = [AD[x] for x in raw_actions]

    text = text.lower().strip()
    for raw_action, text_action in zip(raw_actions, text_actions):
        text = text.replace(text_action, raw_action)
    object_names = [x.lower() for x in LD.values()]

    # ----------- GET PLAN FROM TEXT ----------- #
    plan = ""
    readable_plan = ""
    lines = [line.strip() for line in text.split("\n")]
    for line in lines:
        if '[COST]' in line:
            break
        # Extracting actions
        action_list = [action in line.split() for action in raw_actions]
        if sum(action_list) == 0:
            continue
        action = raw_actions[np.where(action_list)[0][0]]
        # Extracting Objects
        n_objs = len(actions_params_dict[action].parameters.vars())
        objs = get_ordered_objects(object_names, line)
        if len(objs) != n_objs:
            continue
        readable_objs = [obj.replace(' block', '') for obj in objs]
        objs = [BD[x] for x in objs]
        readable_action = "({} {})".format(action, " ".join(readable_objs[:n_objs + 1]))
        if not ground_flag:
            action = "({} {})".format(action, " ".join(objs[:n_objs + 1]))
        else:
            action = "({}_{})".format(action, "_".join(objs[:n_objs + 1]))

        plan += f"{action}\n"
        readable_plan += f"{readable_action}\n"
    # print(f"[+]: Saving plan in {plan_file}")
    file = open(plan_file, "wt")
    file.write(plan)
    file.close()

    return plan, readable_plan
