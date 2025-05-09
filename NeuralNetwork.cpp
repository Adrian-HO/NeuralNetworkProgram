#include <iostream>
#include <thread>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <limits>
#include <omp.h>  // Include OpenMP header

using namespace std;

// Structure to hold wine data
struct WineData {
    vector<double> features;
    double quality;
};

class NeuralNetwork {
private:
    // Network architecture
    int input_size;
    int hidden1_size;
    int hidden2_size;
    int output_size;

    // Learning rate
    double learning_rate;
    double initial_learning_rate;

    // Weights
    vector<vector<double>> weights_input_hidden1;   // weights from input to first hidden layer
    vector<vector<double>> weights_hidden1_hidden2; // weights from first hidden to second hidden layer
    vector<double> weights_hidden2_output;          // weights from second hidden layer to output

    // Activation function: sigmoid
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // Derivative of sigmoid
    double sigmoid_derivative(double x) {
        return x * (1.0 - x);
    }
    
    // ReLU activation function
    double relu(double x) {
        return max(0.0, x);
    }
    
    // Derivative of ReLU
    double relu_derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    // Batch normalization
    vector<double> batch_norm(const vector<double>& inputs, double epsilon = 1e-5) {
        if (inputs.empty()) return inputs;
        
        // Calculate mean
        double mean = 0.0;
        for (double val : inputs) {
            mean += val;
        }
        mean /= inputs.size();
        
        // Calculate variance
        double variance = 0.0;
        for (double val : inputs) {
            variance += (val - mean) * (val - mean);
        }
        variance /= inputs.size();
        
        // Normalize
        vector<double> normalized(inputs.size());
        for (size_t i = 0; i < inputs.size(); i++) {
            normalized[i] = (inputs[i] - mean) / sqrt(variance + epsilon);
        }
        
        return normalized;
    }

    // Random number generator for weight initialization
    mt19937 rng;
    uniform_real_distribution<double> dist;

    // Flag for parallelization
    bool use_parallel;
    int num_threads;

public:
    // Regularization parameter
    double l2_reg_lambda = 0.001;

    // Constructor
    NeuralNetwork(int input_size, int hidden1_size, int hidden2_size, int output_size, 
                 double learning_rate = 0.01, bool use_parallel = false, int num_threads = 4) 
        : input_size(input_size), hidden1_size(hidden1_size), hidden2_size(hidden2_size), 
          output_size(output_size), learning_rate(learning_rate), initial_learning_rate(learning_rate),
          use_parallel(use_parallel), num_threads(num_threads) {
        
        // Set number of threads if parallel execution is enabled
        if (use_parallel) {
            omp_set_num_threads(num_threads);
        }
        
        // Initialize random number generator
        random_device rd;
        rng = mt19937(rd());
        dist = uniform_real_distribution<double>(-0.1, 0.1);  // Smaller initial weights
        
        // Initialize weights for input to first hidden layer
        weights_input_hidden1.resize(input_size, vector<double>(hidden1_size));
        for (int i = 0; i < input_size; i++) {
            for (int h = 0; h < hidden1_size; h++) {
                weights_input_hidden1[i][h] = dist(rng); // Random initialization
            }
        }
        
        // Initialize weights for first hidden to second hidden layer
        weights_hidden1_hidden2.resize(hidden1_size, vector<double>(hidden2_size));
        for (int h1 = 0; h1 < hidden1_size; h1++) {
            for (int h2 = 0; h2 < hidden2_size; h2++) {
                weights_hidden1_hidden2[h1][h2] = dist(rng);
            }
        }
        
        // Initialize weights for second hidden layer to output
        weights_hidden2_output.resize(hidden2_size);
        for (int h = 0; h < hidden2_size; h++) {
            weights_hidden2_output[h] = dist(rng);
        }
    }

    // Adjust learning rate based on epoch
    void adjust_learning_rate(int epoch) {
        if (epoch % 200 == 0 && epoch > 0) {
            learning_rate *= 0.8;  // Reduce by 20% every 200 epochs
            cout << "Learning rate adjusted to: " << learning_rate << endl;
        }
    }

    // Forward pass through the network
    double forward_pass(const vector<double>& inputs, 
                      vector<double>& hidden1_outputs, 
                      vector<double>& hidden2_outputs) {
        // Calculate outputs from first hidden layer using ReLU
        hidden1_outputs.resize(hidden1_size);
        
        if (use_parallel) {
            // Parallel implementation for first hidden layer
            #pragma omp parallel for
            for (int h = 0; h < hidden1_size; h++) {
                double sum = 0.0;
                for (int i = 0; i < input_size; i++) {
                    sum += inputs[i] * weights_input_hidden1[i][h];
                }
                hidden1_outputs[h] = relu(sum);  // Using ReLU
            }
            
            // Apply batch normalization
            if (hidden1_outputs.size() > 1) {
                vector<double> normalized = batch_norm(hidden1_outputs);
                hidden1_outputs = normalized;
            }
            
            // Calculate outputs from second hidden layer
            hidden2_outputs.resize(hidden2_size);
            
            #pragma omp parallel for
            for (int h2 = 0; h2 < hidden2_size; h2++) {
                double sum = 0.0;
                for (int h1 = 0; h1 < hidden1_size; h1++) {
                    sum += hidden1_outputs[h1] * weights_hidden1_hidden2[h1][h2];
                }
                hidden2_outputs[h2] = relu(sum);  // Using ReLU
            }
            
            // Apply batch normalization
            if (hidden2_outputs.size() > 1) {
                vector<double> normalized = batch_norm(hidden2_outputs);
                hidden2_outputs = normalized;
            }
            
            // Calculate output from output layer
            double final_input = 0.0;
            
            #pragma omp parallel reduction(+:final_input)
            {
                #pragma omp for
                for (int h = 0; h < hidden2_size; h++) {
                    final_input += hidden2_outputs[h] * weights_hidden2_output[h];
                }
            }
            
            double final_output = sigmoid(final_input);  // Keep sigmoid for final output
            return final_output;
        } 
        else {
            // Sequential implementation
            for (int h = 0; h < hidden1_size; h++) {
                double sum = 0.0;
                for (int i = 0; i < input_size; i++) {
                    sum += inputs[i] * weights_input_hidden1[i][h];
                }
                hidden1_outputs[h] = relu(sum);  // Using ReLU
            }
            
            // Apply batch normalization
            if (hidden1_outputs.size() > 1) {
                vector<double> normalized = batch_norm(hidden1_outputs);
                hidden1_outputs = normalized;
            }
            
            // Calculate outputs from second hidden layer
            hidden2_outputs.resize(hidden2_size);
            for (int h2 = 0; h2 < hidden2_size; h2++) {
                double sum = 0.0;
                for (int h1 = 0; h1 < hidden1_size; h1++) {
                    sum += hidden1_outputs[h1] * weights_hidden1_hidden2[h1][h2];
                }
                hidden2_outputs[h2] = relu(sum);  // Using ReLU
            }
            
            // Apply batch normalization
            if (hidden2_outputs.size() > 1) {
                vector<double> normalized = batch_norm(hidden2_outputs);
                hidden2_outputs = normalized;
            }
            
            // Calculate output from output layer
            double final_input = 0.0;
            for (int h = 0; h < hidden2_size; h++) {
                final_input += hidden2_outputs[h] * weights_hidden2_output[h];
            }
            double final_output = sigmoid(final_input);  // Keep sigmoid for final output
            
            return final_output;
        }
    }

    // Backward pass to update weights
    double backward_pass(const vector<double>& inputs, 
                       const vector<double>& hidden1_outputs,
                       const vector<double>& hidden2_outputs,
                       double final_output, double target) {
        // Calculate error
        double error = target - final_output;
        
        // Calculate gradients for output layer weights
        double d_output = error * sigmoid_derivative(final_output);
        
        if (use_parallel) {
            // Update output layer weights in parallel with L2 regularization
            #pragma omp parallel for
            for (int h = 0; h < hidden2_size; h++) {
                weights_hidden2_output[h] += learning_rate * (hidden2_outputs[h] * d_output - l2_reg_lambda * weights_hidden2_output[h]);
            }
            
            // Calculate gradients for second hidden layer
            vector<double> d_hidden2(hidden2_size);
            
            #pragma omp parallel for
            for (int h = 0; h < hidden2_size; h++) {
                d_hidden2[h] = weights_hidden2_output[h] * d_output * relu_derivative(hidden2_outputs[h]);
            }
            
            // Update weights between first and second hidden layers with L2 regularization
            #pragma omp parallel for collapse(2)
            for (int h1 = 0; h1 < hidden1_size; h1++) {
                for (int h2 = 0; h2 < hidden2_size; h2++) {
                    weights_hidden1_hidden2[h1][h2] += learning_rate * 
                        (hidden1_outputs[h1] * d_hidden2[h2] - l2_reg_lambda * weights_hidden1_hidden2[h1][h2]);
                }
            }
            
            // Calculate gradients for first hidden layer
            vector<double> d_hidden1(hidden1_size, 0.0);
            
            #pragma omp parallel for
            for (int h1 = 0; h1 < hidden1_size; h1++) {
                double gradient_sum = 0.0;
                for (int h2 = 0; h2 < hidden2_size; h2++) {
                    gradient_sum += weights_hidden1_hidden2[h1][h2] * d_hidden2[h2];
                }
                d_hidden1[h1] = gradient_sum * relu_derivative(hidden1_outputs[h1]);
            }
            
            // Update weights between input and first hidden layer with L2 regularization
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < input_size; i++) {
                for (int h = 0; h < hidden1_size; h++) {
                    weights_input_hidden1[i][h] += learning_rate * 
                        (inputs[i] * d_hidden1[h] - l2_reg_lambda * weights_input_hidden1[i][h]);
                }
            }
        }
        else {
            // Sequential implementation with L2 regularization
            // Update output layer weights
            for (int h = 0; h < hidden2_size; h++) {
                weights_hidden2_output[h] += learning_rate * 
                    (hidden2_outputs[h] * d_output - l2_reg_lambda * weights_hidden2_output[h]);
            }
            
            // Calculate gradients for second hidden layer
            vector<double> d_hidden2(hidden2_size);
            for (int h = 0; h < hidden2_size; h++) {
                d_hidden2[h] = weights_hidden2_output[h] * d_output * relu_derivative(hidden2_outputs[h]);
            }
            
            // Update weights between first and second hidden layers
            for (int h1 = 0; h1 < hidden1_size; h1++) {
                for (int h2 = 0; h2 < hidden2_size; h2++) {
                    weights_hidden1_hidden2[h1][h2] += learning_rate * 
                        (hidden1_outputs[h1] * d_hidden2[h2] - l2_reg_lambda * weights_hidden1_hidden2[h1][h2]);
                }
            }
            
            // Calculate gradients for first hidden layer
            vector<double> d_hidden1(hidden1_size);
            for (int h1 = 0; h1 < hidden1_size; h1++) {
                d_hidden1[h1] = 0.0;
                for (int h2 = 0; h2 < hidden2_size; h2++) {
                    d_hidden1[h1] += weights_hidden1_hidden2[h1][h2] * d_hidden2[h2];
                }
                d_hidden1[h1] *= relu_derivative(hidden1_outputs[h1]);
            }
            
            // Update weights between input and first hidden layer
            for (int i = 0; i < input_size; i++) {
                for (int h = 0; h < hidden1_size; h++) {
                    weights_input_hidden1[i][h] += learning_rate * 
                        (inputs[i] * d_hidden1[h] - l2_reg_lambda * weights_input_hidden1[i][h]);
                }
            }
        }
        
        return error;
    }

    // Train the network with mini-batch support and early stopping
    void train(const vector<vector<double>>& training_inputs, 
              const vector<double>& training_outputs, int epochs, 
              int batch_size = 32, bool verbose = true) {
        if (verbose) {
            cout << "Training the network for " << epochs << " epochs on " 
                 << training_inputs.size() << " samples..." << endl;
            cout << "Using " << (use_parallel ? "parallel" : "sequential") << " execution";
            if (use_parallel) cout << " with " << num_threads << " threads";
            cout << endl;
        }
        
        // Create indices for shuffling
        vector<size_t> indices(training_inputs.size());
        for (size_t i = 0; i < training_inputs.size(); i++) {
            indices[i] = i;
        }
        
        // Early stopping parameters
        double best_mse = numeric_limits<double>::max();
        int patience = 50;  // Number of epochs to wait for improvement
        int counter = 0;
        
        // Keep track of validation error
        vector<vector<double>> validation_inputs;
        vector<double> validation_outputs;
        
        // Split off 10% of training data for validation
        size_t validation_size = training_inputs.size() / 10;
        if (validation_size > 0) {
            // Shuffle indices
            shuffle(indices.begin(), indices.end(), rng);
            
            // Take first 10% for validation
            for (size_t i = 0; i < validation_size; i++) {
                validation_inputs.push_back(training_inputs[indices[i]]);
                validation_outputs.push_back(training_outputs[indices[i]]);
            }
            
            // Remove validation data from training indices
            indices.erase(indices.begin(), indices.begin() + validation_size);
        }
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Adjust learning rate based on epoch
            adjust_learning_rate(epoch);
            
            double total_error = 0.0;
            
            // Shuffle indices for stochastic gradient descent
            shuffle(indices.begin(), indices.end(), rng);
            
            // Process mini-batches
            size_t num_batches = (indices.size() + batch_size - 1) / batch_size;
            
            for (size_t batch = 0; batch < num_batches; batch++) {
                size_t start_idx = batch * batch_size;
                size_t end_idx = min(start_idx + batch_size, indices.size());
                
                // Process each sample in the batch
                if (use_parallel) {
                    #pragma omp parallel
                    {
                        #pragma omp for reduction(+:total_error) schedule(dynamic)
                        for (size_t i = start_idx; i < end_idx; i++) {
                            size_t idx = indices[i];
                            vector<double> hidden1_outputs;
                            vector<double> hidden2_outputs;
                            
                            // Forward pass
                            double final_output = forward_pass(training_inputs[idx], 
                                                            hidden1_outputs, hidden2_outputs);
                            
                            // Backward pass
                            double error = backward_pass(training_inputs[idx], 
                                                      hidden1_outputs, hidden2_outputs,
                                                      final_output, training_outputs[idx]);
                            
                            total_error += error * error; // squared error
                        }
                    }
                } else {
                    for (size_t i = start_idx; i < end_idx; i++) {
                        size_t idx = indices[i];
                        vector<double> hidden1_outputs;
                        vector<double> hidden2_outputs;
                        
                        // Forward pass
                        double final_output = forward_pass(training_inputs[idx], 
                                                        hidden1_outputs, hidden2_outputs);
                        
                        // Backward pass
                        double error = backward_pass(training_inputs[idx], 
                                                  hidden1_outputs, hidden2_outputs,
                                                  final_output, training_outputs[idx]);
                        
                        total_error += error * error; // squared error
                    }
                }
            }
            
            // Calculate mean squared error on training data
            double train_mse = total_error / indices.size();
            
            // Calculate validation error if validation data exists
            double val_mse = 0.0;
            if (!validation_inputs.empty()) {
                double val_error = 0.0;
                for (size_t i = 0; i < validation_inputs.size(); i++) {
                    vector<double> hidden1_outputs;
                    vector<double> hidden2_outputs;
                    double prediction = forward_pass(validation_inputs[i], hidden1_outputs, hidden2_outputs);
                    double error = validation_outputs[i] - prediction;
                    val_error += error * error;
                }
                val_mse = val_error / validation_inputs.size();
                
                // Check for early stopping based on validation error
                if (val_mse < best_mse) {
                    best_mse = val_mse;
                    counter = 0;
                } else {
                    counter++;
                    if (counter >= patience) {
                        if (verbose) {
                            cout << "Early stopping at epoch " << epoch << endl;
                            cout << "Best validation MSE: " << best_mse << endl;
                        }
                        break;  // Stop training
                    }
                }
            } else {
                // If no validation data, use training error for early stopping
                if (train_mse < best_mse) {
                    best_mse = train_mse;
                    counter = 0;
                } else {
                    counter++;
                    if (counter >= patience) {
                        if (verbose) {
                            cout << "Early stopping at epoch " << epoch << endl;
                            cout << "Best training MSE: " << best_mse << endl;
                        }
                        break;  // Stop training
                    }
                }
            }
            
            // Print progress periodically
            if (verbose && (epoch % 100 == 0 || epoch == epochs - 1)) {
                cout << "Epoch " << epoch << ": Train MSE = " << fixed << setprecision(6) << train_mse;
                if (!validation_inputs.empty()) {
                    cout << ", Val MSE = " << fixed << setprecision(6) << val_mse;
                }
                cout << endl;
            }
        }
        
        // Reset learning rate to initial value
        learning_rate = initial_learning_rate;
    }

    // Make a prediction
    double predict(const vector<double>& inputs) {
        vector<double> hidden1_outputs;
        vector<double> hidden2_outputs;
        return forward_pass(inputs, hidden1_outputs, hidden2_outputs);
    }

    // Evaluate on test dataset
    double evaluate(const vector<vector<double>>& test_inputs, 
                  const vector<double>& test_outputs) {
        double total_error = 0.0;
        double total_abs_error = 0.0;
        
        if (use_parallel) {
            #pragma omp parallel
            {
                double local_error = 0.0;
                double local_abs_error = 0.0;
                
                #pragma omp for schedule(dynamic)
                for (size_t i = 0; i < test_inputs.size(); i++) {
                    double prediction = predict(test_inputs[i]);
                    double error = test_outputs[i] - prediction;
                    local_error += error * error;
                    local_abs_error += fabs(error);
                }
                
                #pragma omp critical
                {
                    total_error += local_error;
                    total_abs_error += local_abs_error;
                }
            }
        } else {
            for (size_t i = 0; i < test_inputs.size(); i++) {
                double prediction = predict(test_inputs[i]);
                double error = test_outputs[i] - prediction;
                total_error += error * error;
                total_abs_error += fabs(error);
            }
        }
        
        double mse = total_error / test_inputs.size();
        double mae = total_abs_error / test_inputs.size();
        
        cout << "Test set evaluation:\n";
        cout << "  Mean Squared Error: " << fixed << setprecision(6) << mse << endl;
        cout << "  Mean Absolute Error: " << fixed << setprecision(6) << mae << endl;
        
        // Print detailed prediction statistics
        vector<double> predictions;
        predictions.reserve(test_inputs.size());
        
        for (size_t i = 0; i < test_inputs.size(); i++) {
            predictions.push_back(predict(test_inputs[i]));
        }
        
        // Calculate min, max, mean, and variance of predictions
        double min_pred = *min_element(predictions.begin(), predictions.end());
        double max_pred = *max_element(predictions.begin(), predictions.end());
        
        double sum_pred = 0.0;
        for (double pred : predictions) {
            sum_pred += pred;
        }
        double mean_pred = sum_pred / predictions.size();
        
        double variance_pred = 0.0;
        for (double pred : predictions) {
            variance_pred += (pred - mean_pred) * (pred - mean_pred);
        }
        variance_pred /= predictions.size();
        double std_dev_pred = sqrt(variance_pred);
        
        cout << "Prediction statistics:\n";
        cout << "  Min: " << fixed << setprecision(6) << min_pred << endl;
        cout << "  Max: " << fixed << setprecision(6) << max_pred << endl;
        cout << "  Mean: " << fixed << setprecision(6) << mean_pred << endl;
        cout << "  Standard Deviation: " << fixed << setprecision(6) << std_dev_pred << endl;
        
        return mse;
    }
};

// Function to load wine data from CSV
vector<WineData> load_wine_data(const string& filename) {
    vector<WineData> data;
    ifstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }
    
    string line;
    // Skip header line
    getline(file, line);
    
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        WineData wine;
        
        // Parse the line
        for (int i = 0; i < 12; i++) {  // 11 features + 1 quality value
            if (!getline(ss, cell, ';')) {
                cerr << "Error parsing line: " << line << endl;
                break;
            }
            
            // Remove quotes if present
            if (!cell.empty() && cell.front() == '"' && cell.back() == '"') {
                cell = cell.substr(1, cell.size() - 2);
            }
            
            double value;
            try {
                value = stod(cell);
            } catch (const exception& e) {
                cerr << "Error converting to number: " << cell << endl;
                continue;
            }
            
            if (i < 11) {
                wine.features.push_back(value);
            } else {
                wine.quality = value;
            }
        }
        
        if (wine.features.size() == 11) {
            data.push_back(wine);
        }
    }
    
    file.close();
    cout << "Loaded " << data.size() << " wine samples from " << filename << endl;
    return data;
}

// Function to normalize features
void normalize_features(vector<WineData>& data) {
    if (data.empty()) return;
    
    int num_features = data[0].features.size();
    vector<double> min_values(num_features, numeric_limits<double>::max());
    vector<double> max_values(num_features, numeric_limits<double>::lowest());
    
    // Find min and max values for each feature
    for (const auto& wine : data) {
        for (int i = 0; i < num_features; i++) {
            min_values[i] = min(min_values[i], wine.features[i]);
            max_values[i] = max(max_values[i], wine.features[i]);
        }
    }
    
    // Normalize features to [0, 1]
    for (auto& wine : data) {
        for (int i = 0; i < num_features; i++) {
            if (max_values[i] > min_values[i]) {
                wine.features[i] = (wine.features[i] - min_values[i]) / (max_values[i] - min_values[i]);
            } else {
                wine.features[i] = 0.5; // If all values are the same
            }
        }
    }
    
    // Normalize quality to [0.1, 0.9] instead of [0, 1] for better gradient flow
    double min_quality = numeric_limits<double>::max();
    double max_quality = numeric_limits<double>::lowest();
    
    for (const auto& wine : data) {
        min_quality = min(min_quality, wine.quality);
        max_quality = max(max_quality, wine.quality);
    }
    
    for (auto& wine : data) {
        // Scale to [0.1, 0.9] range
        wine.quality = 0.1 + 0.8 * (wine.quality - min_quality) / (max_quality - min_quality);
    }
    
    cout << "Features normalized to [0, 1] range" << endl;
    cout << "Quality range: [" << min_quality << ", " << max_quality << "] normalized to [0.1, 0.9]" << endl;
}

// Function to split data into training and testing sets
void split_data(const vector<WineData>& data, double train_ratio,
               vector<vector<double>>& train_inputs, vector<double>& train_outputs,
               vector<vector<double>>& test_inputs, vector<double>& test_outputs) {
    
    // Create a copy of indices and shuffle them
    vector<size_t> indices(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        indices[i] = i;
    }
    
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);
    
    // Calculate split point
    size_t train_size = static_cast<size_t>(data.size() * train_ratio);
    
    // Resize vectors to accommodate data
    train_inputs.resize(train_size);
    train_outputs.resize(train_size);
    test_inputs.resize(data.size() - train_size);
    test_outputs.resize(data.size() - train_size);
    
    // Split the data according to shuffled indices
    for (size_t i = 0; i < train_size; i++) {
        train_inputs[i] = data[indices[i]].features;
        train_outputs[i] = data[indices[i]].quality;
    }
    
    for (size_t i = train_size; i < data.size(); i++) {
        test_inputs[i - train_size] = data[indices[i]].features;
        test_outputs[i - train_size] = data[indices[i]].quality;
    }
    
    cout << "Data split into " << train_inputs.size() << " training samples and "
         << test_inputs.size() << " testing samples" << endl;
}

// Function to run performance benchmarks
void run_benchmarks(const vector<vector<double>>& train_inputs, const vector<double>& train_outputs,
                    const vector<vector<double>>& test_inputs, const vector<double>& test_outputs) {
    
    cout << "\n====== PERFORMANCE BENCHMARKS ======\n";
    cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << " threads\n";
    
    // Test parameters
    const int epochs = 100;
    const int batch_size = 32;
    const int hidden1_size = 16;  // Increased from 8
    const int hidden2_size = 8;   // Increased from 4
    const double learning_rate = 0.05;
    
    // Vector to store timing results
    vector<pair<string, double>> timing_results;
    
    // Sequential benchmark
    {
        cout << "\nRunning sequential benchmark...\n";
        NeuralNetwork nn(11, hidden1_size, hidden2_size, 1, learning_rate, false);
        
        auto start_time = chrono::high_resolution_clock::now();
        nn.train(train_inputs, train_outputs, epochs, batch_size, false);
        auto end_time = chrono::high_resolution_clock::now();
        
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        double seconds = duration.count() / 1000.0;
        
        cout << "Sequential execution completed in " << seconds << " seconds\n";
        nn.evaluate(test_inputs, test_outputs);
        
        timing_results.push_back({"Sequential", seconds});
    }
    
    // Test different thread counts
    vector<int> thread_counts = {2, 4, 8, 16};
    for (int threads : thread_counts) {
        if (threads > std::thread::hardware_concurrency()) {
            cout << "\nSkipping " << threads << " threads (exceeds hardware concurrency)\n";
            continue;
        }
        
        cout << "\nRunning parallel benchmark with " << threads << " threads...\n";
        NeuralNetwork nn(11, hidden1_size, hidden2_size, 1, learning_rate, true, threads);
        
        auto start_time = chrono::high_resolution_clock::now();
        nn.train(train_inputs, train_outputs, epochs, batch_size, false);
        auto end_time = chrono::high_resolution_clock::now();
        
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        double seconds = duration.count() / 1000.0;
        
        cout << "Parallel execution with " << threads << " threads completed in " 
             << seconds << " seconds\n";
        nn.evaluate(test_inputs, test_outputs);
        
        timing_results.push_back({"Parallel (" + to_string(threads) + " threads)", seconds});
    }
    
    // Report speedup
    cout << "\n====== SPEEDUP SUMMARY ======\n";
    cout << "| Configuration | Time (s) | Speedup |\n";
    cout << "|---------------|----------|--------|\n";
    
    double base_time = timing_results[0].second;
    for (const auto& result : timing_results) {
        double speedup = base_time / result.second;
        cout << "| " << left << setw(13) << result.first << " | " 
             << fixed << setprecision(4) << setw(8) << result.second << " | " 
             << fixed << setprecision(2) << speedup << "x |\n";
    }
    
    // Experiment with different batch sizes (using best thread count)
    if (!timing_results.empty() && timing_results.size() > 1) {
        // Find best thread count
        int best_thread_count = 2;  // Default
        double best_time = numeric_limits<double>::max();
        
        for (size_t i = 1; i < timing_results.size(); i++) {
            if (timing_results[i].second < best_time) {
                best_time = timing_results[i].second;
                string config = timing_results[i].first;
                size_t pos = config.find("(") + 1;
                best_thread_count = stoi(config.substr(pos));
            }
        }
        
        cout << "\n====== BATCH SIZE EXPERIMENT ======\n";
        cout << "Using " << best_thread_count << " threads (best performance)\n";
        
        vector<int> batch_sizes = {1, 8, 32, 64, 128, 256};
        vector<pair<int, double>> batch_timing;
        
        for (int batch : batch_sizes) {
            cout << "\nTesting batch size " << batch << "...\n";
            NeuralNetwork nn(11, hidden1_size, hidden2_size, 1, learning_rate, true, best_thread_count);
            
            auto start_time = chrono::high_resolution_clock::now();
            nn.train(train_inputs, train_outputs, epochs, batch, false);
            auto end_time = chrono::high_resolution_clock::now();
            
            auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
            double seconds = duration.count() / 1000.0;
            
            cout << "Completed in " << seconds << " seconds\n";
            batch_timing.push_back({batch, seconds});
        }
        
        cout << "\n| Batch Size | Time (s) |\n";
        cout << "|------------|----------|\n";
        for (const auto& result : batch_timing) {
            cout << "| " << left << setw(10) << result.first << " | " 
                 << fixed << setprecision(4) << result.second << " |\n";
        }
    }
}

int main() {
    // Load wine quality data
    vector<WineData> wine_data = load_wine_data("winequality-white.csv");
    
    if (wine_data.empty()) {
        cerr << "No data loaded. Exiting." << endl;
        return 1;
    }
    
    // Normalize features
    normalize_features(wine_data);
    
    // Split data into training and testing sets (80% training, 20% testing)
    vector<vector<double>> train_inputs, test_inputs;
    vector<double> train_outputs, test_outputs;
    split_data(wine_data, 0.8, train_inputs, train_outputs, test_inputs, test_outputs);
    
    // Run performance benchmarks
    run_benchmarks(train_inputs, train_outputs, test_inputs, test_outputs);
    
    // Final training with best configuration
    cout << "\n====== FINAL TRAINING ======\n";
    
    // Create a neural network with parallelization
    int num_threads = min(8, (int)thread::hardware_concurrency());
    NeuralNetwork nn(11, 16, 8, 1, 0.05, true, num_threads);
    nn.l2_reg_lambda = 0.001;  // Add regularization
    
    // Train with more epochs for final model
    int epochs = 2000;
    int batch_size = 64;  // Optimal batch size from experiment
    
    cout << "Training final model with " << num_threads << " threads, " 
         << epochs << " epochs, batch size " << batch_size << "...\n";
    
    auto start_time = chrono::high_resolution_clock::now();
    nn.train(train_inputs, train_outputs, epochs, batch_size);
    auto end_time = chrono::high_resolution_clock::now();
    
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    cout << "\nTraining completed in " << duration.count() / 1000.0 << " seconds" << endl;
    
    // Evaluate on test set
    cout << "\nEvaluating final model on test set...\n";
    nn.evaluate(test_inputs, test_outputs);
    
    return 0;
}