use ndarray::prelude::*;
use ndarray::Array1;

#[derive(Debug)]
struct Input{
    xs: Array1<f64>,
    ys: Array1<f64>
}


#[derive(Debug)]
struct Model{
    input: Input,
    yhat: Array1<f64>,
    weight: Array1<f64>,
    bias: f64
}

impl Model {
    fn new(shape: usize)-> Self{
        Self{
            yhat: Array1::zeros(shape),
            weight: Array1::zeros(shape),
            bias: 0.0,
            input: Input{xs: Array1::zeros(shape), ys: Array1::zeros(shape)} 
        }
    }

    fn initialize(xs: Array1<f64>, ys: Array1<f64>)->Self{
        Self{Input{xs,
        ys
        }
    }
    }

    fn dot_product(&self) -> f64{
        self.weight.dot(&self.input.xs) + self.bias
    }

    fn sigmoid(result: f64) -> f64 {
        1.0 / (1.0 + (-result).exp())
    }

    //fn loss()
}

fn main() {
    println!("Hello, world!");
    let xs = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let ys = array![2.0, 4.0, 6.0, 8.0, 10.0];
    let shape = xs.shape()[0];
    let model = Model::new(shape);
    model::initialize(xs, ys);
    model.dot_product();

    println!("{:?} and {:?}",model.weight, model.bias);
}
