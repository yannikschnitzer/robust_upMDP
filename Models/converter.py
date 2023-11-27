import Markov.models as Markov
import stormpy
import stormpy.core
import stormpy.logic
import stormpy.pars
import stormpy.examples
import stormpy.examples.files


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
        complicated_ind = model.Labels.index("(((x > (15 - 2)) & (y > (15 - 2))) & (z > (15 - 2)))")
        model.Labels[complicated_ind] = "reached"
        model.Labels.pop(-1)
        model.opt = "max"
        model.Formulae = ["Pmax=? [F \"reached\"]"]
    elif 'coin' in filename:
        reached_states = set(model.Labelled_states[model.Labels.index("finished")]).intersection(
                             model.Labelled_states[model.Labels.index("all_coins_equal_1")])
        model.Labelled_states.append(list(reached_states))
    elif 'brp_256' in filename:
        model.Labelled_states.append(model.Labelled_states[model.Labels.index("(s = 5)")])
    elif 'crowds' in filename:
        model.Labelled_states.append(model.Labelled_states[model.Labels.index("observe0Greater1")])
    elif 'nand' in filename:
        model.Labelled_states.append(model.Labelled_states[model.Labels.index("target")])

    if "min" in model.Formulae[0]:
        model.opt = "min"
    model.paramed = [[[not t.value().is_constant() for t in action.transitions]
                       for action in state.actions]
                       for state in storm_model.states]
    if str(model.model.model_type)=='ModelType.DTMC':
        instantiator = stormpy.pars.PDtmcInstantiator(model.model)
    elif  str(model.model.model_type)=='ModelType.MDP':
        instantiator = stormpy.pars.PMdpInstantiator(model.model)
    else:
        raise RuntimeError("Invalid model type (should be a pDTMC or pMDP).")
    sample_params = model.param_sampler()
    sample = instantiator.instantiate(sample_params)
   
    model.Transition_probs = [[[t.value() if not model.paramed[s_id][a_id][t_id] else 'p' 
                                for t_id, t in enumerate(action.transitions)]
                                for a_id, action in enumerate(state.actions)]
                                for s_id, state in enumerate(sample.states)]
    model.max_supports = sum([sum([any([not t.value().is_constant() for t in action.transitions])
                        for action in state.actions])
                        for state in storm_model.states])
    model.paramed_states = [i for i, states in enumerate(model.paramed) if any([any(acts) for acts in states])]
    
    return model
