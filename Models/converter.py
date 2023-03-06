import Markov.models as Markov


def parse(storm_model, params, filename, props, f, weather= None):
    model = Markov.storm_upMDP()
    model.model =  storm_model
    model.params = params
    model.filename = filename
    model.weather = weather
    model.props = props
    
    model.Labels = []
    model.Labelled_states = []
   
    model.States = []
    model.Actions = []
    model.Enabled_actions = []
    model.trans_ids = []
    
    for state in storm_model.states:
        state_ids = []
        model.States.append(int(state))
        for label in state.labels:
            if str(label) in model.Labels:
                i = model.Labels.index(str(label))
                model.Labelled_states[i].append(int(state))
            else:
                model.Labels.append(str(label))
                model.Labelled_states.append([int(state)])
        enabled_acts = []
        for action in state.actions:
            act_ids = []
            if int(str(action)) not in model.Actions:
                model.Actions.append(int(str(action)))
            enabled_acts.append(int(str(action)))
            for transition in action.transitions:
                act_ids.append(int(str(transition).split(',')[0].split('(')[-1]))
            state_ids.append(act_ids)
        model.trans_ids.append(state_ids)
        model.Enabled_actions.append(enabled_acts)
    model.Init_state = model.Labelled_states[model.Labels.index("init")][0]
    model.Formulae = [f]
    if 'reached' not in model.Labels:
        model.Labels.append("reached")
    if 'drone' in filename:
        model.Labelled_states.append(
                model.Labelled_states[
                    model.Labels.index("(((x > (15 - 2)) & (y > (15 - 2))) & (z > (15 - 2)))")])
        model.opt = "max"
        model.Formulae = ["Pmax=? [F \"reached\"]"]
    elif 'coin' in filename:
        reached_states = set(model.Labelled_states[model.Labels.index("finished")]).intersection(
                             model.Labelled_states[model.Labels.index("all_coins_equal_1")])
        model.Labelled_states.append(list(reached_states))
        model.opt = "min"
    elif 'brp_256' in filename:
        model.Labelled_states.append(model.Labelled_states[model.Labels.index("(s = 5)")])
    elif 'crowds' in filename:
        model.Labelled_states.append(model.Labelled_states[model.Labels.index("observe0Greater1")])
    elif 'nand' in filename:
        model.opt="min"
        model.Labelled_states.append(model.Labelled_states[model.Labels.index("target")])
    
    model.paramed = [[[not t.value().is_constant() for t in action.transitions]
                       for action in state.actions]
                       for state in storm_model.states]
    model.max_supports = sum([sum([any([not t.value().is_constant() for t in action.transitions])
                        for action in state.actions])
                        for state in storm_model.states])
    return model
