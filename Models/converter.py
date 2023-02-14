import Markov.models as Markov


def parse(storm_model, params, filename, props, weather= None):
    model = Markov.storm_upMDP()
    model.model =  storm_model
    model.params = params
    model.filename = filename
    model.weather = weather
    model.props = props
    model.Init_state = 0
    return model
