import cntk

# The data in the file must satisfied the following format:
# |labels 0 0 1 |features 2.1 7.0 2.2 - the format consist of 4 features and one 3 component hot vector
#represents the iris flowers
def create_reader(path, is_training, input_dim, num_label_classes):
    # create 2 streams for the data
    labelStream = cntk.io.StreamDef(field='label', shape = num_label_classes, is_sparse=False)
    featureStream = cntk.io.StreamDef(field='features', shape=input_dim, is_sparse=False)

    #create deserializer by providing the file path, and related streams
    deserailizer = cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(labels = labelStream, features = featureStream))

    #create mini batch source as function return
    mb = cntk.io.MinibatchSource(deserailizer, randomize = is_training, max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)
    return mb

# create a model - FFNN 4-50-3
# i.e 4 inputs, then 50 hidden nodes then 3 outputs
# which are one-ho
def create_model(features, hid_dim, out_dim):
    # perform initialization
    with cntk.layers.default_options(init=cntk.glorot_uniform()):
        #hidden layer with hid_def number of neurons and tanh activation function
        h1=cntk.layers.Dense(hid_dim, activation= cntk.ops.tanh, name='hidLayer')(features)

        #output layer with out_dim neurons
        o = cntk.layers.Dense(out_dim, activation = None)(h1)
        return o

# Monitor training progress
def print_training_progress(trainer, mb, frequency):
    t_loss = "NA"
    e_error  = "NA"

    if mb % frequency == 0:
        t_loss = trainer.previous_minibatch_loss_average
        e_error = trainer.previous_minibatch_evaluation_average
        print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, t_loss, e_error*100))
    return mb, t_loss, e_error

# Set up NN
input_dim = 4
hidden_dim = 50
num_output_classes = 3
input = cntk.input_variable(input_dim)
label = cntk.input_variable(num_output_classes)

# create a reader to read from the file
reader_train = create_reader("D:/Users/Sachit/source/repos/SamplesRepo/IrisData/IrisData/iris-data/trainData_cntk.txt", True, input_dim, num_output_classes)

# Create the model
z = create_model(input, hidden_dim, num_output_classes)
loss = cntk.cross_entropy_with_softmax(z, label)
label_error = cntk.classification_error(z, label)

learning_rate = 0.2
lr_schedule = cntk.learning_parameter_schedule(learning_rate)
learner = cntk.sgd(z.parameters, lr_schedule)
trainer = cntk.Trainer(z, (loss, label_error), [learner])

#Init the params for trainer
minibatch_size = 120 
num_iterations = 20

# Map the data streams to input and labels
input_map = {label: reader_train.streams.labels , input: reader_train.streams.features}

training_output_freq = 2
plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0, int(num_iterations)):
    # Read a mb
    data = reader_train.next_minibatch(minibatch_size, input_map = input_map)
    trainer.train_minibatch(data)
    batchsize, loss, error = print_training_progress(trainer, i, training_output_freq)
    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)

# plot data
import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["loss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')
 
#plt.show()
  
plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["error"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error')
#plt.show()

# Now look at testing data
reader_test = create_reader("D:/Users/Sachit/source/repos/SamplesRepo/IrisData/IrisData/iris-data/testData_cntk.txt", False, input_dim, num_output_classes)
test_input_map = {
    label: reader_test.streams.labels,
    input: reader_test.streams.features
    }

# Test data for trained mode
test_mini_size = 20
num_samples = 20
num_mbs = num_samples // test_mini_size
test_result = 0.0

for i in range(num_mbs):
    data = reader_test.next_minibatch(test_mini_size, input_map = test_input_map)
    eval_error = trainer.test_minibatch(data)
    test_result = test_result + eval_error

# Average of evaluation errors of all test minibatches
z.save("D:/Users/Sachit/source/repos/SamplesRepo/IrisData/IrisData/iris-data/saved_model.model") 
print("Average test error: {0:.2f}%".format(test_result*100 / num_mbs))
