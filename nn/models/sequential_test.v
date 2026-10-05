module models

import vtl.nn.types

fn test_nnc() {
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
	nn.input([1, 2])
	nn.sigmoid()
	assert nn.info.layers.len == 2
	assert nn.info.layers[0].output_shape() == [1, 2]
	assert nn.info.layers[1].output_shape() == [1, 2]
}

fn test_nn() {
	mut nn := sequential_with_layers[f64]([]types.Layer[f64]{})
}

fn test_added_activation_layers_in_sequential() {
	mut nn := sequential[f64]()
	nn.input([3])
	nn.softplus()
	nn.selu()
	nn.hardswish()
	assert nn.info.layers.len == 4
	assert nn.info.layers[1].output_shape() == [3]
	assert nn.info.layers[2].output_shape() == [3]
	assert nn.info.layers[3].output_shape() == [3]
}
