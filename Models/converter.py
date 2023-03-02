import Markov.models as Markov


def parse(storm_model, params, filename, props, f, weather= None):
    model = Markov.storm_upMDP()
    model.model =  storm_model
    model.params = params
    model.filename = filename
    model.weather = weather
    model.props = props
    model.Init_state = 0
    
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
    model.Formulae = [f] 

    if 'drone' in filename:
        model.Labels[-1] = 'reached'
    else:
        model.Labels.append("reached")
        reached_states = set(model.Labelled_states[1]).intersection(model.Labelled_states[2])
        model.Labelled_states.append(list(reached_states))
        model.opt = "min"

    return model
