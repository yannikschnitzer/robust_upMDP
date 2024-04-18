import pickle
import stormpy 

def load_samples(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data[0], data[1]

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def gen_samples(model, N):
    print("Num Samples:", N)
    sample_array = [[0.869661074152221, 0.8661798728546303],
[0.8628228067490115, 0.7951723834808841],
[0.8620823631217877, 0.8714771565000735],
[0.9125711960410308, 0.7818002418429729],
[0.8664964122259171, 0.8960449627795338],
[0.8207954625322589, 0.8478324770294388],
[0.8446490767938636, 0.7811642917262241],
[0.9191153719297983, 0.7936198699381651],
[0.8864324321419775, 0.9398784811009875],
[0.8072730371156094, 0.9214250405778299],
[0.8420246149869187, 0.8972084444641392],
[0.8142367177230883, 0.7645572236541063],
[0.8535789801814365, 0.9292351353230426],
[0.9404609671356791, 0.8981776326384185],
[0.9389832926382342, 0.8688531847967902],
[0.7788434399690806, 0.7766300925600227],
[0.8655612875208566, 0.807913759226833],
[0.9128951254260605, 0.9446908388879199],
[0.9181904445042588, 0.8186386387017405],
[0.8728675186314087, 0.8039202096577984],
[0.8141614815010048, 0.9416018043926462],
[0.894056609631649, 0.7803382945204862],
[0.9462064773805487, 0.9432385806262048],
[0.8662302942971775, 0.8359354135017256],
[0.9454742779252265, 0.9327566945388481],
[0.7891022521435348, 0.7940675455636694],
[0.8478332331535391, 0.8464875591470622],
[0.7560019550295863, 0.9075757972906222],
[0.919733042652475, 0.8571301792542154],
[0.776367088979513, 0.888508061753783],
[0.8539588632425581, 0.9460041241195614],
[0.8174110654953042, 0.839610145980437],
[0.7809625724275472, 0.7883420718045608],
[0.8962950766442683, 0.7802926338459671],
[0.758057517063453, 0.9448165582495263],
[0.8863605052592239, 0.9435688529419254],
[0.8966147221187213, 0.9269095061000869],
[0.9144542603216151, 0.9354983889793231],
[0.8665000316611954, 0.8093972753701186],
[0.8211059514261604, 0.7949595888993074],
[0.7773432476768609, 0.9001380274688975],
[0.9312926571008756, 0.7691136733637],
[0.777831935573643, 0.9455121616980932],
[0.9407532471901455, 0.8287119787494865],
[0.93707401731815, 0.849280773022506],
[0.86129116040779, 0.7874370910427937],
[0.8761749704651074, 0.9132910983465249],
[0.8882308317040237, 0.7607575455866953],
[0.8035269051102294, 0.7703719057570444],
[0.7541695967217074, 0.8023149107472405],
[0.8739084093192329, 0.8729259607404543],
[0.8153935594189515, 0.8896511402792422],
[0.9393455984400545, 0.8251406907963449],
[0.7808023107849057, 0.7961230804185307],
[0.7887863970815752, 0.8024022572165067],
[0.8379140887768536, 0.7821401352947365],
[0.8839915545670884, 0.7884123168258372],
[0.7564994535991607, 0.8664001552947421],
[0.790647594031505, 0.7786707605674618],
[0.7576181222118429, 0.8965876175117424],
[0.7765583541314293, 0.8484018585740156],
[0.9243102287160047, 0.929037329965087],
[0.8028733313825461, 0.8636652873622469],
[0.8999197414186468, 0.9481558393427307],
[0.7696627449350127, 0.8263074789200674],
[0.7710043977756637, 0.9089318698534582],
[0.7937505668413533, 0.7865729594308228],
[0.7640131064330025, 0.8696561703568575],
[0.8945685836323555, 0.9316092560536077],
[0.8983882218249633, 0.7970101459490133],
[0.9428958868773423, 0.8637631187357111],
[0.8334767460907865, 0.8719718042570912],
[0.8876999313960762, 0.8194797983401028],
[0.8226111348988409, 0.8025265578131713],
[0.9178967584965603, 0.847781213628355],
[0.9449466771184044, 0.7579594678705677],
[0.9386995544070273, 0.7667129265196834],
[0.8241194113323729, 0.8795446187165159],
[0.9463263502874981, 0.8468138470835735],
[0.8625146375357325, 0.8937363884968075],
[0.7701891105662485, 0.832576694671406],
[0.8209130323908724, 0.7565063523344151],
[0.8182881359632435, 0.9060313064004152],
[0.7555076790747178, 0.8912587528616311],
[0.9093872151640465, 0.9209969356933501],
[0.8877620330564763, 0.8477963836333269],
[0.8046465003298993, 0.8922137366504654],
[0.9273053342594724, 0.9060979512743106],
[0.8617416922353869, 0.9104776817067612],
[0.8940385658512139, 0.8762068778469483],
[0.9260996696841426, 0.7730672997136568],
[0.7928506371236378, 0.9043207731805663],
[0.9467322177479449, 0.8681645840398681],
[0.7828209550843966, 0.8852096013957035],
[0.8582511552774076, 0.8326364975739124],
[0.752511256424224, 0.9023799177444063],
[0.8059996330397992, 0.8144577025710942],
[0.8639259500921146, 0.7699413793481303],
[0.8078359945966669, 0.8142189406199101],
[0.8599469592329363, 0.8339209513787446],
[0.8439581629653166, 0.7545798442215862],
[0.9435254672479564, 0.9058091800642971],
[0.9452580074115017, 0.8321888066785246],
[0.8316396741950082, 0.947412065426984],
[0.9110468007749386, 0.9435067056378407],
[0.7630040546447122, 0.9036210643669036],
[0.8285943647484781, 0.9441352730163515],
[0.7957424125696898, 0.888502815089333],
[0.7897761016379974, 0.8006650937316652],
[0.86971331409188, 0.7701492820342618],
[0.9042236713512435, 0.9247220731595525],
[0.8279032850499705, 0.8710180899780273],
[0.8937266185220625, 0.9408054716676756],
[0.8922795664828006, 0.8030317335893044],
[0.819233840415852, 0.8326745635854406],
[0.7717208501239421, 0.7904936568376514],
[0.8396552864129448, 0.8374550488312702],
[0.8431209728244962, 0.7989180548468384],
[0.9307660013928662, 0.8123004417052305],
[0.8914870811492165, 0.7841593743130052],
[0.802865930474326, 0.8853577026004155],
[0.9208303925136925, 0.8037666152088834],
[0.8474507181556294, 0.8209504212152183],
[0.7548741715115243, 0.7912764784152361],
[0.8160001847596308, 0.8783069728059263],
[0.9350801956224486, 0.8841986022336942],
[0.8880926763698689, 0.8238250852119715],
[0.7932214307662371, 0.917735069861121],
[0.8214051237349826, 0.7687783466681211],
[0.926705321207683, 0.8488155088002795],
[0.7927602380634285, 0.8475342007970494],
[0.8370785492484717, 0.8588181468019249],
[0.9372337159020414, 0.8257671361892905],
[0.8056207849969788, 0.8983862415876168],
[0.8714359730574621, 0.9338147021557929],
[0.7746614259359873, 0.8840383692510128],
[0.8474345194987345, 0.7572381702602125],
[0.8792299299550719, 0.8338101180165431],
[0.8253845071433555, 0.8848136422269776],
[0.9308768951650631, 0.9379538257109541],
[0.8757867489381924, 0.8135935244839961],
[0.7939796305339843, 0.9007619778241038],
[0.9263938756428617, 0.9413422946158977],
[0.8145724829085756, 0.8608690732459168],
[0.7541369963633324, 0.9385646018725706],
[0.862718742621092, 0.7999256838689162],
[0.8890466969037458, 0.7855077550415062],
[0.931667586330991, 0.7833276484843958],
[0.9449938785384322, 0.8839347740080574],
[0.8005159965770425, 0.8417222457879142],
[0.8211773558585393, 0.9091281777765773],
[0.8118221819975167, 0.7983023789126061],
[0.8286368532220029, 0.8544988108877223],
[0.941593635808627, 0.8744388276919854],
[0.8876714809062609, 0.8393379859407071],
[0.8462994341003872, 0.864989642769751],
[0.9148954081870773, 0.8996654088825253],
[0.860033035521423, 0.8646648869319089],
[0.7727639563987678, 0.7627892611405721],
[0.7846783691742896, 0.9481139381979343],
[0.8897549549500728, 0.918109022676225],
[0.7669085102199967, 0.8589416768099017],
[0.8759917226991576, 0.9398555812818074],
[0.7880729067075148, 0.8466489025073481],
[0.7569456020621416, 0.7918772644214528],
[0.9065056086754775, 0.9055313975210852],
[0.8269575986433032, 0.750244546319593],
[0.7865134970574614, 0.8158611436996182],
[0.7846700997910284, 0.8163601162128511],
[0.8242801719835136, 0.7652122177693648],
[0.7540614156248427, 0.8614252473612652],
[0.9381475949823159, 0.9365830146791285],
[0.8176271981025773, 0.8425272176235751],
[0.82921427908884, 0.8900054140891325],
[0.792354648960877, 0.8612769144176942],
[0.8379152246678843, 0.8356206878248577],
[0.7644866516553773, 0.797839449073275],
[0.8232255148418921, 0.7987409114120468],
[0.7656748730115985, 0.9463650324894134],
[0.7733559286363932, 0.8112037214473087],
[0.9209721479903581, 0.9022411219476942],
[0.920285617908467, 0.828367434657395],
[0.82821622720122, 0.9072511270115743],
[0.7752032625389197, 0.8673823640443569],
[0.83870840007941, 0.9456389291827728],
[0.92790492422636, 0.8899135474347133],
[0.9006921732198278, 0.9022424889721989],
[0.8604231714316336, 0.8112788728172164],
[0.8999886272607708, 0.8432740193874557],
[0.8136387303852035, 0.9080058189296076],
[0.7862467827075921, 0.7609881077830323],
[0.9153524557311132, 0.8170990968306981],
[0.8086187148990875, 0.9353933497082121],
[0.9292719225879763, 0.8728503697294773],
[0.9012708342108429, 0.8093270449645616],
[0.9165936013541574, 0.9108105580655912],
[0.856994281268246, 0.9476266134542404],
[0.815338044744243, 0.8835840331897347],
[0.8062789035771472, 0.8567954586099776],
[0.8373221598856115, 0.8793932673362972]
]
    
    samples = []

    for pair in sample_array:
        #print("Pair:",pair)
        point = dict()
        i = 0
        for e in model.params:
            point[e] = stormpy.RationalRF(pair[i])
            #print(e, pair[i])
            i += 1
        rational_parameter_assignments = dict(
            [[x, val] for x, val in point.items()])
        samples.append(rational_parameter_assignments)
    #print("Hallo")
    samples = [model.param_sampler() for j in range(N)]
    return samples

def get_samples(args):
    if args["sample_load_file"] is not None:
        samples = load_samples(args["sample_load_file"])
    else:
        samples = gen_samples(args["model"], args["num_samples"])
    if args["sample_save_file"] is not None:
        save_data(args["sample_save_file"], samples)
    return samples
