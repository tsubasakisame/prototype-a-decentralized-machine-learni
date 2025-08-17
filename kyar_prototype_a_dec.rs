use serde::{Serialize, Deserialize};
use tokio::{task, sync::mpsc};
use rand::Rng;
use ndarray::Array2;
use ndarray_linalg::{Eigenvalues, SingularValues};

// Node configuration
const NODE_COUNT: usize = 5;
const EPOCHS: usize = 10;
const BATCH_SIZE: usize = 32;

// Data structure for the model
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Model {
    weights: Array2<f64>,
    bias: Array2<f64>,
}

// Data structure for the data
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Data {
    inputs: Array2<f64>,
    labels: Array2<f64>,
}

// Function to generate random data
fn generate_data() -> (Data, Data) {
    let mut rng = rand::thread_rng();
    let input_dim = 10;
    let output_dim = 2;
    let sample_count = 100;

    let inputs = Array2::from_shape_vec((sample_count, input_dim), (0..sample_count * input_dim).map(|_| rng.gen_range(0.0..1.0)).collect()).unwrap();
    let labels = Array2::from_shape_vec((sample_count, output_dim), (0..sample_count * output_dim).map(|_| rng.gen_range(0.0..1.0)).collect()).unwrap();

    let (train_inputs, test_inputs) = inputs.split_at(Axis(0), sample_count * 4 / 5);
    let (train_labels, test_labels) = labels.split_at(Axis(0), sample_count * 4 / 5);

    (
        Data {
            inputs: train_inputs.clone(),
            labels: train_labels.clone(),
        },
        Data {
            inputs: test_inputs.clone(),
            labels: test_labels.clone(),
        },
    )
}

// Function to train the model
async fn train_model(mut node_rx: mpsc::Receiver<Model>, mut node_tx: mpsc::Sender<Model>) {
    let (data, _) = generate_data();
    let mut model = Model {
        weights: Array2::zeros((data.inputs.ncols(), 2)),
        bias: Array2::zeros((2,)),
    };

    for _ in 0..EPOCHS {
        // Send the current model to all nodes
        for _ in 0..NODE_COUNT {
            node_tx.send(model.clone()).await.unwrap();
        }

        // Receive the updated models from the nodes
        let mut updated_models = vec![];
        for _ in 0..NODE_COUNT {
            updated_models.push(node_rx.recv().await.unwrap());
        }

        // Aggregate the updated models
        let mut aggregated_weights = Array2::zeros((data.inputs.ncols(), 2));
        let mut aggregated_bias = Array2::zeros((2,));
        for updated_model in updated_models {
            aggregated_weights += &updated_model.weights;
            aggregated_bias += &updated_model.bias;
        }
        aggregated_weights /= NODE_COUNT as f64;
        aggregated_bias /= NODE_COUNT as f64;

        model.weights = aggregated_weights.clone();
        model.bias = aggregated_bias.clone();
    }
}

// Function to run a node
async fn run_node(mut node_rx: mpsc::Receiver<Model>, mut node_tx: mpsc::Sender<Model>, mut data_rx: mpsc::Receiver<Data>) {
    let mut local_model = Model {
        weights: Array2::zeros((10, 2)),
        bias: Array2::zeros((2,)),
    };

    while let Some(data) = data_rx.recv().await {
        // Get the current model from the aggregator
        let mut model = node_rx.recv().await.unwrap();

        // Compute the gradients
        let mut weights_grad = Array2::zeros((10, 2));
        let mut bias_grad = Array2::zeros((2,));

        // Compute the gradients for this batch
        for i in 0..BATCH_SIZE {
            let input = data.inputs.slice(s![i, ..]);
            let label = data.labels.slice(s![i, ..]);

            let output = input.dot(&model.weights) + &model.bias;
            let error = &output - &label;
            let weights_grad_batch = input.t().dot(&error);
            let bias_grad_batch = error.clone();

            weights_grad += &weights_grad_batch;
            bias_grad += &bias_grad_batch;
        }

        weights_grad /= BATCH_SIZE as f64;
        bias_grad /= BATCH_SIZE as f64;

        // Update the local model
        local_model.weights -= &weights_grad;
        local_model.bias -= &bias_grad;

        // Send the updated local model to the aggregator
        node_tx.send(local_model.clone()).await.unwrap();
    }
}

#[tokio::main]
async fn main() {
    // Create channels for communication between nodes
    let (node_tx, node_rx) = mpsc::unbounded_channel();
    let (data_tx, data_rx) = mpsc::unbounded_channel();

    // Create the aggregator task
    task::spawn(train_model(node_rx.clone(), node_tx.clone()));

    // Create node tasks
    for _ in 0..NODE_COUNT {
        task::spawn(run_node(node_rx.clone(), node_tx.clone(), data_rx.clone()));
    }

    // Send the data to the nodes
    for _ in 0..EPOCHS {
        let (data, _) = generate_data();
        data_tx.send(data).await.unwrap();
    }
}