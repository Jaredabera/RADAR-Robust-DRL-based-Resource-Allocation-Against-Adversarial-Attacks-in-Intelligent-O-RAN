import tensorflow as tf
import numpy as np

def create_student_model(input_shape, output_size):
    """Create a student model matching the teacher's architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(output_size)
    ])
    return model

def generate_synthetic_data(input_shape, num_samples=10000):
    """Generate synthetic data matching the input shape."""
    return np.random.random((num_samples,) + input_shape)

def defensive_distillation(teacher_model_path, temperature=10, num_epochs=100, batch_size=32):
    # Load the teacher model
    teacher_model = tf.saved_model.load(teacher_model_path)
    
    # Print available signatures
    print("Available signatures:", list(teacher_model.signatures.keys()))
    
    # Use the 'action' signature
    action_fn = teacher_model.signatures['action']
    
    # Print the input and output specs
    print("Input specs:", action_fn.structured_input_signature)
    print("Output specs:", action_fn.structured_outputs)
    
    # Define input shape and output size based on the model specs
    input_specs = action_fn.structured_input_signature[1]
    input_shape = input_specs['0/observation'].shape[1:]
    output_size = action_fn.structured_outputs['action'].shape[-1]
    
    print("Input shape:", input_shape)
    print("Output size:", output_size)
    
    # Create and compile the student model
    student_model = create_student_model(input_shape, output_size)
    student_model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Generate synthetic training data
    X_train = generate_synthetic_data(input_shape)
    
    # Prepare input for the teacher model
    inputs = {
        '0/discount': tf.constant([1.0] * X_train.shape[0], dtype=tf.float32),
        '0/observation': tf.constant(X_train, dtype=tf.float32),
        '0/reward': tf.constant([0.0] * X_train.shape[0], dtype=tf.float32),
        '0/step_type': tf.constant([0] * X_train.shape[0], dtype=tf.int32)
    }
    
    # Get soft labels from the teacher model
    teacher_predictions = action_fn(**inputs)
    logits = teacher_predictions['action'] / temperature
    soft_labels = tf.nn.softmax(logits, axis=-1)
    
    # Train the student model on soft labels
    student_model.fit(X_train, soft_labels, epochs=num_epochs, batch_size=batch_size, verbose=1)
    
    return student_model

# Usage
teacher_model_path = 'D:/DRL based Resource Allocation in/New RA/colosseum-oran-commag-dataset-main/ml_models/embb_policy'
distilled_model = defensive_distillation(teacher_model_path)

# Save the distilled model
distilled_model.save('distilled_model')