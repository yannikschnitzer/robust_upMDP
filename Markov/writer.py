import os
import logging
from tqdm import tqdm
import math
import numpy as np
import stormpy
import time

def dec_round(num, decimals):
    """
    Decimal rounding, lowerbounds at 10^-decimals
    """
    power = 10**decimals
    return max(10**-decimals,math.trunc(power*num)/power)

class stormpy_io:
    def __init__(self, _model, _N=1, _horizon="infinite"):
        self.model = _model
        self.N = _N
        if _horizon != "infinite":
            self.horizon = str(N) + "_steps"
        else:
            self.horizon = _horizon
       
        self.opt_thresh = True

    def read(self):
        pass

    def _write_labels(self):
        labels = self.model.Labels
        state_labeling = stormpy.StateLabeling(len(self.model.States))

        for label in labels:
            state_labeling.add_label(label)
        for label, subset in zip(self.model.Labels, self.model.Labelled_states):    
            state_labeling.set_states(label, stormpy.BitVector(len(self.model.States),subset))
        return state_labeling

    def _write_transitions(self):
        if self.model.Enabled_actions is not None:
            builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=True, row_groups=0)
        else:
            builder = stormpy.SparseMatrixBuilder(rows=0, columns=0, entries=0, force_dimensions=False, has_custom_row_grouping=False)
        choices = 0
        for s in self.model.States:
            self_loop = True
            if self.model.Enabled_actions is not None:
                builder.new_row_group(choices)
                for a in self.model.Enabled_actions[s]:
                    for i, s_prime in enumerate(self.model.trans_ids[s][a]):
                        self_loop = False
                        builder.add_next_value(choices, s_prime, self.model.Transition_probs[s][a][i])
                    choices += 1
                if self_loop:
                    builder.add_next_value(choices, s, 1)
                    choices += 1
            else:
                for i, s_prime in enumerate(self.model.trans_ids[s]):
                    self_loop = False
                    builder.add_next_value(s, s_prime, self.model.Transition_probs[s][i])
                if self_loop:
                    builder.add_next_value(s, s, 1)
        return builder.build() 
        
    def write(self):
        if hasattr(self.model, "mdp"):
            self.mdp = self.model.mdp
            self.specs = self.model.props
        else:
            labels = self._write_labels()
            trans_mat = self._write_transitions()  
            components = stormpy.SparseModelComponents(trans_mat, labels)
            self.mdp = stormpy.storage.SparseMdp(components)
            self.specs = self._write_specification()
        

    def _write_specification(self):
        specs = []
        for f in self.model.Formulae:
            specs += stormpy.parse_properties_without_context(f)
        return specs

    def solve(self):
        if hasattr(self.model, "mdp"):
            res = []
            all_res = []
            for spec in self.specs:
                result = stormpy.model_checking(self.mdp, spec, extract_scheduler=True)
                if self.model.Actions is not None:
                    pol = np.zeros((len(self.model.States), len(self.model.Actions)))
                    for s_num, s in enumerate(self.mdp.states):
                        choice = result.scheduler.get_choice(s)
                        act = choice.get_deterministic_choice()
                        pol[s_num, act]  = 1
                else:
                    pol = None
                all_res += [result.at(state) for state in self.model.States]
                res.append(result.at(self.mdp.initial_states[0]))
        else: 
            res = []
            all_res = []
            for spec in self.specs:
                result = stormpy.check_model_sparse(self.mdp, spec, extract_scheduler=True)
                if self.model.Actions is not None:
                    pol = np.zeros((len(self.model.States), len(self.model.Actions)))
                    for s_num, s in enumerate(self.mdp.states):
                        choice = result.scheduler.get_choice(s)
                        act = choice.get_deterministic_choice()
                        pol[s_num, act]  = 1
                else:
                    pol = None
                all_res += [result.at(state) for state in self.model.States]
                res.append(result.at(self.mdp.initial_states[0]))
        return res, all_res, pol

class PRISM_io:
    """
    Class for creating PRISM files
    """
    def __init__(self, _model, _N=-1, input_folder='input', output_folder='output', _explicit=True, _horizon="infinite"):
        self.model = _model
        self.explicit = _explicit
        self.N = _N
        if _horizon != "infinite":
            self.horizon = str(N) + "_steps"
        else:
            self.horizon = _horizon
       
        in_folder = input_folder + "/" + _model.Name + "_" + _horizon 
        out_folder = output_folder + "/" + _model.Name + "_" + _horizon 
        input_prefix = in_folder + "/model" 
        
        if not os.path.exists(out_folder):
            os.makedirs(out_folder+'/')
        if not os.path.exists(in_folder):
            os.makedirs(in_folder+'/')
        
        self.prism_filename = input_prefix + ".prism"
        self.spec_filename = input_prefix + ".pctl"
        
        self.state_filename = input_prefix + ".sta"
        self.label_filename = input_prefix + ".lab"
        self.transition_filename = input_prefix + ".tra"
        self.all_filename = input_prefix + ".all"

        file_prefix = out_folder + "/PRISM_out"
        self.vector_filename = file_prefix + "_vector.csv"
        if self.model.Enabled_actions is not None:
            self.policy_filename = file_prefix + "_policy.csv"
        self.opt_thresh = True
        self.thresh = 0.5
        self.max = True
        self.spec = "until"

    def write(self):
        if self.explicit:
            self._write_explicit()
        else:
            self._write()

    def write_file(self, content, filename, mode="w"):
        """
        function for writing to a file
        """
        filehandle = open(filename, mode)
        filehandle.writelines(content)
        filehandle.close()

    def read(self):
        """
        Reads the results of solving the prism files from earlier
        """
        if self.model.Enabled_actions is not None:
            policy_file = self.policy_filename
            vector_file = self.vector_filename
            policy = np.genfromtxt(policy_file, delimiter=',', dtype='str')
            if self.horizon != 'infinite':
                policy = np.flipud(policy)

                if len(np.shape(policy)) > 1:
                    optimal_policy= np.zeros(np.shape(policy)+tuple([2]))
                    optimal_reward = np.zeros(np.shape(policy)[1])
                    print(np.shape(optimal_policy))
                else:
                    optimal_policy= np.zeros(tuple([1])+np.shape(policy))
                    optimal_reward = np.zeros(np.shape(policy)[0])

                optimal_reward = np.genfromtxt(vector_file).flatten()
                if not self.opt_thresh:
                    if self.max:
                        optimal_reward = optimal_reward >= self.thresh
                    else:
                        optimal_reward = optimal_reward <= self.thresh


                if len(np.shape(policy)) > 1:
                    for i, row in enumerate(policy):
                        for j, value in enumerate(row):
                            if value != '':
                                value_split = value.split('_')
                                optimal_policy[i,j] = int(value_split[-1])
                            else:
                                optimal_policy[i,j] = -1
                else:
                    i = 0
                    for j, value in enumerate(policy):
                        if value != '':
                            value_split = value.split('_')
                            optimal_policy[i,j] = int(value_split[-1])
                        else:
                            optimal_policy[i,j] = -1
            else:
                if len(np.shape(policy)) > 1:
                    optimal_policy= np.zeros(np.shape(policy)+tuple([2]))
                    optimal_reward = np.zeros(np.shape(policy)[1])
                    print(np.shape(optimal_policy))
                else:
                    optimal_policy= np.zeros(np.shape(self.model.States))
                    optimal_reward = np.zeros(np.shape(policy)[0])

                optimal_reward = np.genfromtxt(vector_file).flatten()
                if not self.opt_thresh:
                    if self.max:
                        optimal_reward = optimal_reward >= self.thresh
                    else:
                        optimal_reward = optimal_reward <= self.thresh
                    
                for i, row in enumerate(policy):
                    if i > 0:
                        if row != '':
                            row_split = row.split(' ')
                            optimal_policy[int(row_split[0])] = int(row_split[-1].split('_')[-1])
                        else:
                            optimal_policy[i] = -1
        else:
            optimal_policy = None
            vector_file = self.vector_filename
            optimal_reward = np.genfromtxt(vector_file).flatten()
            if not self.opt_thresh:
                if self.max:
                    optimal_reward = optimal_reward >= self.thresh
                else:
                    optimal_reward = optimal_reward <= self.thresh

        return optimal_policy, optimal_reward
    
    def _write_labels(self):
        
        labels = self.model.Labels
        # label_file_list = ['0="init" 1="deadlock" 2="reached" 3="critical"']
        #label_file_list = ['#DECLARATION',' '.join([label for i, label in enumerate(labels)]),'#END']
        label_file_list = [str(i) + '="' + label + '"' for i, label in enumerate(labels)]
        label_file_list = [' '.join(label_file_list)]

        m = self.model
        substring = ['' for i in m.States]
        for i, s in enumerate(m.States):
            substring[i] = str(i) + ":"

        labelled = [False for s in substring]
        for i, subset in enumerate(m.Labelled_states):
            for s in subset:
                substring[s] += ' '+str(i)
                labelled[s] = True
        substring = [elem for i, elem in enumerate(substring) if labelled[i]]
        label_file_list += substring

        label_string = '\n'.join(label_file_list)

        self.write_file(label_string, self.label_filename)

    def _write_states(self):
        state_list = ['(x)']
        counter=0
        
        m = self.model
        state_list += [str(i)+':('+str(i)+')'\
                         for i in range(len(m.States))]
        
        state_string = '\n'.join(state_list)
        
        self.write_file(state_string, self.state_filename)

    def _write_transitions(self):
        m = self.model
       
        if m.Enabled_actions is not None: 
            nr_choices_absolute = 0
            nr_transitions_absolute = 0
            transition_file_list = []
            
            transition_file_list_states = ['' for i in m.States]
            for i, s in enumerate(m.States):
                choice = 0
                selfloop = False
                if len(m.Enabled_actions[i]) > 0:
                    subsubstring = ['' for j in m.Enabled_actions[i]]

                    for a_idx, a in enumerate(m.Enabled_actions[i]):
                        action_label = "a_"+str(a)
                        substring_start = str(i) + ' '+str(choice)
                        
                        prob_idxs = [j for j in m.trans_ids[i][a]]
                       
                        trans_strings = [str(dec_round(prob,6))
                                                    for prob in m.Transition_probs[i][a]]
                        
                        subsublist = [substring_start+" "+str(j)+" "+prob+" "+action_label
                                            for (j, prob) in zip(prob_idxs, trans_strings)]

                        choice += 1
                        nr_choices_absolute += 1
                        nr_transitions_absolute += len(subsublist)
                        subsubstring[a_idx] = '\n'.join(subsublist)
                else:
                    if not selfloop:
                        subsubstring = []
                        action_label = "a_" + str(i)
                        subsubstring += [str(i) +' ' + str(choice)+ ' ' +str(i)+\
                                         ' 1.0  '+ action_label]
                        nr_choices_absolute += 1
                        choice += 1

                        selfloop = True

                        nr_transitions_absolute += len(subsubstring)
                    else:
                            subsubstring = []
                substring = [subsubstring]
                transition_file_list_states[i] = substring
            
            del(subsubstring)
            del(substring)
            flatten = lambda t: [item for sublist in t
                                      for subsublist in sublist
                                      for item in subsublist]
            size_states = len(m.States)
            size_choices = nr_choices_absolute
            size_transitions = nr_transitions_absolute

            model_size = {'States': size_states,
                          'Choices': size_choices,
                          'Transitions':size_transitions}
            header = str(size_states)+' '+str(size_choices)+' '+str(size_transitions)+'\n'
            #header = 'MdP\n'

            self.write_file(header, self.transition_filename)
            for sublist in transition_file_list_states:
                for subsublist in sublist:
                    for item in subsublist:
                        self.write_file(item+'\n', self.transition_filename, 'a')
        else:
            nr_transitions_absolute = 0
            transition_file_list = []
            
            transition_file_list_states = ['' for i in m.States]
            for i, s in enumerate(m.States):
                
                prob_idxs = [j for j in m.trans_ids[i]]
                
                trans_strings = [str(dec_round(prob,6))
                                            for prob in m.Transition_probs[i]]
                
                subsublist = [str(i)+" "+str(j)+" "+prob
                                    for (j, prob) in zip(prob_idxs, trans_strings)]

                nr_transitions_absolute += len(subsublist)
                transition_file_list.append('\n'.join(subsublist))
            nr_states = len(m.States)
            #nr_states = m.States.size
            header = str(nr_states) + " " + str(nr_transitions_absolute) + '\n'
            self.write_file(header, self.transition_filename)
            self.write_file('\n'.join(transition_file_list), self.transition_filename, 'a')

    def _write_explicit(self):
        
        self._write_states()
        self._write_labels()
        self._write_transitions()

        self.specification = self.writePRISM_specification()

    def _write(self):
        raise NotImplementedError

    def writePRISM_specification(self):
        """
        Writes PRISM specification file in PCTL
        """
        N = self.N
        model = self.model
        horizon = self.horizon
        if horizon != "infinite":
            horizonLen = int(N)
            specification = "Pmax=? [ F<="+str(horizonLen)+' "reached" ]'
            #specification = "Pmaxmin=? [ F<="+str(horizonLen)+' "reached" ]'
        else:
            if self.max:
                specification = 'Pmax=? [ F "reached" ]'
            else:
                specification = 'Pmin=? [ F "reached" ]'
        self.write_file(specification, self.spec_filename)
        return specification

    def solve(self,java_memory=8, prism_folder="~/code/prism-imc/prism"):
        """
        function for solving iMDP using PRISM
        """
        import subprocess
        spec = self.specification
    
        if self.model.Enabled_actions is not None:
            options = ' -ex -exportadv "'+self.policy_filename+'"' + \
                      ' -exportvector "'+self.vector_filename+'"'
        else:
            options = ""
            #options = ' -ex -exportvector "'+self.vector_filename+'"'


        if self.explicit:
            model_file = '"'+self.all_filename+'"'
            command = prism_folder+"/bin/prism -javamaxmem "+ \
                str(java_memory)+"g -importmodel "+model_file+" -pf '"+ \
                spec+"' "+options
        else:
            prism_file = self.prism_filename
            model_file = '"'+prism_file+'"'

            command = prism_folder + "/bin/prism -javamaxmem " + \
                    str(java_memory) + "g "+model_file+" -pf '"+spec+"' "+options
        pre_command = time.perf_counter()
        res = subprocess.run(command, shell=True, capture_output=True)
        post_command = time.perf_counter()
        if self.model.Enabled_actions is None:
            pol_arr = None
            prob = float(str(res.stdout).split("Result: ")[1].split(" ")[0])
            all_probs = [prob] 
        else:
            pol, all_probs = self.read()
            prob = all_probs[self.model.Init_state]
            pol_arr = np.zeros((len(self.model.States), len(self.model.Actions)))
            for s, a in enumerate(pol):
                pol_arr[s,int(a)] = 1
        post_read = time.perf_counter()
        logging.debug("time solving {}".format(post_command-pre_command))
        logging.debug("time reading {}".format(post_read-post_command))
        return [prob], all_probs, pol_arr
