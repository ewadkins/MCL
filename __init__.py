import numpy as np
from itertools import combinations
import operator as op
import math

class MCL:
    
    def __init__(self, n, dims,
                 translation_magnitude_error=0.2, translation_direction_error=0.1, rotation_error=0.1,
                 use_normal_distribution=True, random_walk=False, random_walk_interval=1,
                 redistribute=True, redistribution_rate=0.03,
                 kld_sampling=True):
        
        self.n = n;
        self.M = n;
        self.dims = dims;
        self.d = len(self.dims);
        self.dims_norm = np.linalg.norm(self.dims);
        self.sigma = self.d * (self.d - 1) / 2; # d choose 2
        self.use_normal_distribution = use_normal_distribution;
        self.random_walk = random_walk;
        self.random_walk_interval = random_walk_interval;
        self.redistribute = redistribute;
        self.redistribution_rate = redistribution_rate;
        self.kld_sampling = kld_sampling;
        self.action_count = 0;
        
        self.translation_magnitude_error = translation_magnitude_error;
        self.translation_direction_error = translation_direction_error;
        self.rotation_error = rotation_error;
        
        self.particles = [Particle(i, self) for i in range(self.M)];
        self.random_initialization();
        
        self.randomize_translation_errors();
        self.randomize_rotation_errors();
        
    def random_initialization(self):
        self.positions = np.random.rand(self.M, self.d) * self.dims;
        self.orientation_offsets = (np.random.rand(self.M, self.sigma) * 2 - 1) * 180;
        self.weights = np.full(self.M, 1.0 / self.M);
        
    def update_weights(self, hypotheses):
        weights = np.zeros((self.M));
        baseline = self.dims_norm;
        
        for hypothesis in hypotheses:
            weights += np.clip(baseline - np.linalg.norm(hypothesis - self.positions, axis=1), 0, baseline) ** 64;
            
        self.weights = weights;
        self.normalize_weights();
            
    def resample(self):
        resample_size = self.n;
        
        variance = self.get_variance();
        if self.kld_sampling:
            resample_size = max(200, int(self.n * min(variance, 1)));
        
        resample = np.random.choice(self.M, resample_size, p=self.weights);
                
        resample_position_variation = 0.001 * variance;
        resample_rotation_variation = 1 * variance;
        
        self.M = resample_size;
        self.positions = self.positions[resample] + (np.random.rand(self.M, self.d) * 2 - 1) * self.dims * resample_position_variation;
        #self.orientation_offsets = (np.random.rand(self.M, self.sigma) * 2 - 1) * 180;
        self.orientation_offsets = self.orientation_offsets[resample] + (np.random.rand(self.M, self.sigma) * 2 - 1) * 180 * resample_rotation_variation;
        self.weights = self.weights[resample];
        self.particles = [Particle(i, self) for i in range(self.M)];

        self.translation_magnitude_errors = self.translation_magnitude_errors[resample];
        self.translation_direction_errors = self.translation_direction_errors[resample];
        self.rotation_errors = self.rotation_errors[resample];
        
        if self.redistribute:
            # Unweighted random choice of particles to redistribute
            redistribution_size = int(self.redistribution_rate * self.M);
            redistributed = np.random.choice(self.M, redistribution_size);
            self.positions[redistributed] = np.random.rand(redistribution_size, self.d) * self.dims;
            self.orientation_offsets[redistributed] = (np.random.rand(redistribution_size, self.sigma) * 2 - 1) * 180;
        
        self.normalize_weights();
    
    def get_variance(self):
        redistribution_size = int(self.redistribution_rate * self.M);
        center_of_mass = np.mean(self.positions, axis=0);
        distances = np.linalg.norm(self.positions - center_of_mass, axis=1);
        if self.redistribute:
            distances = distances[np.argpartition(distances, -redistribution_size)[:-redistribution_size]];
        return np.linalg.norm(distances / self.dims_norm) ** 2;
    
    def normalize_weights(self):
        total = np.sum(self.weights);
        if total == 0:
            self.weights = np.full(self.M, 1.0 / self.M);
        else:
            self.weights = self.weights / total;
        
    def reset_translation_vector(self):
        self.randomized_translation_vector = [0] * self.d;
        
    def reset_rotation_vector(self):
        self.randomized_rotation_vector = [0] * self.sigma;
                
    def check_errors_on_translation(self, translation):
        if self.random_walk:
            if self.action_count % self.random_walk_interval == 0:
                return self.randomize_translation_errors();
        for i in range(self.d):
            if self.randomized_translation_vector[i] != 0 and self.randomized_translation_vector[i] == -np.sign(translation[i]):
                return self.randomize_translation_errors();
            if self.randomized_translation_vector[i] == 0:
                self.randomized_translation_vector[i] = np.sign(translation[i]);
                
    def check_errors_on_rotation(self, rotation):
        if self.random_walk:
            if self.action_count % self.random_walk_interval == 0:
                return self.randomize_rotation_errors();
        for i in range(self.sigma):
            if self.randomized_rotation_vector[i] != 0 and self.randomized_rotation_vector[i] == -np.sign(rotation[i]):
                return self.randomize_rotation_errors();
            if self.randomized_rotation_vector[i] == 0:
                self.randomized_rotation_vector[i] = np.sign(rotation[i]);
        
    def randomize_translation_errors(self):
        #print 'Randomizing translation errors';
        self.reset_translation_vector();
        self.translation_magnitude_errors, self.translation_direction_errors = self.get_random_translation_errors();
        
    def randomize_rotation_errors(self):
        #print 'Randomizing rotation errors';
        self.reset_rotation_vector();
        self.rotation_errors = self.get_random_rotation_errors();
        
    def get_random_translation_errors(self):
        if self.use_normal_distribution:
            translation_magnitude_errors = np.random.normal(scale=self.translation_magnitude_error / 3, size=(self.M, self.d));
            translation_direction_errors = np.random.normal(scale=self.translation_direction_error / 3, size=(self.M, self.sigma)) * 180;
        else:
            translation_magnitude_errors = (np.random.rand(self.M, self.d) * 2 - 1) * self.translation_magnitude_error;
            translation_direction_errors = (np.random.rand(self.M, self.sigma) * 2 - 1) * self.translation_direction_error * 180;
            
        return translation_magnitude_errors, translation_direction_errors;
        
    def get_random_rotation_errors(self):
        if self.use_normal_distribution:
            rotation_errors = np.random.normal(scale=self.rotation_error / 3, size=(self.M, self.sigma));
        else:
            rotation_errors = (np.random.rand(self.M, self.sigma) * 2 - 1) * self.rotation_error;
        return rotation_errors;
    
    def translate(self, *translation):
        self.check_errors_on_translation(translation);
        for particle in self.particles:
            transformed_translation = _rotate_transform(
                *translation, rotation=self.orientation_offsets[particle.id]);
            transformed_translation = _rotate_transform(
                *transformed_translation, rotation=self.translation_direction_errors[particle.id]);
            
            self.positions[particle.id] += transformed_translation * (self.translation_magnitude_errors[particle.id] + 1);
        self.action_count += 1;
            
    def rotate(self, *rotation):
        self.check_errors_on_rotation(rotation);
        for particle in self.particles:
            self.orientation_offsets[particle.id] += rotation * self.rotation_errors[particle.id];
        self.action_count += 1;
    
    def get_particles(self):
        return self.particles;
    
    def get_positions(self):
        return np.array(self.positions);
    
    def get_orientations_offsets(self):
        return np.array(self.orientations_offsets);
    
    def get_weights(self):
        return np.array(self.weights);
    
    def get_min_weight(self):
        return np.min(self.weights);
    
    def get_max_weight(self):
        return np.max(self.weights);

class Particle:
        
    def __init__(self, id, mcl):
        self.id = id;
        self.mcl = mcl;
        
    def get_position(self):
        return self.mcl.positions[self.id];
        
    def get_orientation_offset(self):
        return self.mcl.orientation_offsets[self.id];
        
    def get_weight(self):
        return self.mcl.weights[self.id];
        
    def set_position(self, position):
        self.mcl.positions[self.id] = np.array(position);
        
    def set_orientation_offset(self, orientation_offset):
        self.mcl.orientation_offsets[self.id] = np.array(orientation_offset);
        
    def set_weight(self, weight):
        self.mcl.weights[self.id] = weight;
        
    def __repr__(self):
        return 'Particle(id=' + str(self.id) + ', position='+str(self.get_position())+', orientation_offset='+str(self.get_orientation_offset())+', weight='+str(self.get_weight())+')';

    
def _2_dimensional_rotation_transformation(r):
    return np.array([
            [math.cos(r[0]), -math.sin(r[0])],
            [math.sin(r[0]), math.cos(r[0])]]);

def _3_dimensional_rotation_transformation(r):
    s, c = math.sin, math.cos;
    x, y, z = r[2], r[1], r[0];
    return np.array([
            [c(y)*c(x), c(z)*s(x)*s(y)-c(x)*s(z), c(x)*c(z)*s(y)+s(x)*s(z)],
            [c(y)*s(z), c(x)*c(z)+s(x)*s(y)*s(z), c(x)*s(y)*s(z)-c(z)*s(x)],
            [-s(y), c(y)*s(x), c(x)*c(y)]]);

# n-dimensional rotation transform
def _rotate_transform(*vector, **kwargs):
    d = len(vector);
    sigma = d * (d - 1) / 2;
    rotation = kwargs.get('rotation', None);
    if rotation is None or d == 1:
        return vector;
    if len(rotation) != sigma:
        raise Exception(str(sigma) + ' parameters are required to represent orientation in ' + str(d) + ' dimensions');

    rads = map(lambda x: math.radians(x), rotation);
    transformation = None;
    if d == 2: # 2-dimensional shortcut
        transformation = _2_dimensional_rotation_transformation(rads);
    elif d == 3: # 3-dimensional shortcut
        transformation = _3_dimensional_rotation_transformation(rads);
    else: # n-dimensional generic case
        transformation = np.zeros((d, d));
        np.fill_diagonal(transformation, 1);
        combs = list(combinations(range(d), 2));
        for c in range(sigma):
            i, j = combs[c];
            alt = 1 if c % 2 == 0 else -1;
            m = np.zeros((d, d));
            np.fill_diagonal(m, 1);
            m[i][i] = math.cos(rads[c]);
            m[i][j] = alt * -math.sin(rads[c]);
            m[j][i] = alt * math.sin(rads[c]);
            m[j][j] = math.cos(rads[c]);
            transformation = np.matmul(transformation, m);

    return np.round(np.matmul(vector, transformation), 15);