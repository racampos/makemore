use candle_core::scalar::TensorOrScalar;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::encoding::one_hot;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufRead, BufReader};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "names.txt";
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut words = Vec::new();
    for line in reader.lines() {
        let line = line?;
        words.push(line);
    }

    let words = &words
        .iter()
        .map(|word| format!("{}{}{}", '.', word.to_string(), '.'))
        .collect::<Vec<String>>();

    let N = Tensor::zeros((27, 27), DType::I64, &Device::Cpu)?;

    let mut unique_chars = HashSet::new();
    for word in words {
        for ch in word.chars() {
            unique_chars.insert(ch);
        }
    }
    let mut chars: Vec<char> = unique_chars.into_iter().collect();
    chars.sort();

    let stoi =
        HashMap::<char, i64>::from_iter(chars.iter().enumerate().map(|(i, &c)| (c, i as i64)));

    let itos =
        HashMap::<i64, char>::from_iter(chars.iter().enumerate().map(|(i, &c)| (i as i64, c)));

    let mut b: HashMap<(char, char), i32> = HashMap::new();

    let mut xs: Vec<i64> = Vec::new();
    let mut ys: Vec<i64> = Vec::new();

    for word in words.iter().take(1) {
        for (ch1, ch2) in word.chars().zip(word.chars().skip(1)) {
            let ix1 = stoi.get(&ch1).unwrap();
            let ix2 = stoi.get(&ch2).unwrap();
            let bigram = (ch1, ch2);
            b.insert(bigram, b.get(&bigram).unwrap_or(&0) + 1);
            xs.push(*ix1);
            ys.push(*ix2);
        }
    }

    let W = Tensor::randn(0f32, 1., (27, 27), &Device::Cpu)?;

    let xs = Tensor::from_vec(xs.clone(), xs.clone().len(), &Device::Cpu)?;
    let ys = Tensor::from_vec(ys.clone(), ys.clone().len(), &Device::Cpu)?;
    let xenc = one_hot(xs.clone(), 27, 1f32, 0f32);
    let xenc = xenc.unwrap();
    // println!("{:?}", xenc.to_vec2::<f32>());

    let logits = xenc.matmul(&W)?;
    let counts = logits.exp()?;
    let counts_sum = counts.sum(1)?.unsqueeze(1)?.expand(&[5, 27])?;
    let probs = counts.div(&counts_sum)?;

    // println!("{:?}, shape: {:?}", probs.to_vec2::<f32>(), probs.shape());

    let mut nlls: Vec<Tensor> = Vec::new();
    for i in 0..5 {
        let x = xs.get(i)?.to_vec0::<i64>()?;
        let y = ys.get(i)?.to_vec0::<i64>()?;
        println!("--------");
        println!(
            "bigram example {}: {}{}  (indexes {} {})",
            i + 1,
            itos.get(&x).unwrap(),
            itos.get(&y).unwrap(),
            x,
            y
        );
        println!("input to the neural net: {:?}", x);
        println!(
            "output probabilities from the neural net: {:?}",
            probs.get(i)?.to_vec1::<f32>()?
        );
        println!("label (actual next character): {:?}", y);
        let p = probs.get(i)?.get(y.try_into().unwrap())?;
        println!(
            "probability assigned by the net to the correct character: {:?}",
            p.to_vec0::<f32>()?
        );
        let logp = p.log()?;
        println!("log likelihood: {:?}", logp.to_vec0::<f32>()?);
        let nll = logp.neg()?;
        println!("negative log likelihood: {:?}", nll.to_vec0::<f32>()?);
        nlls.push(nll);
    }
    let nlls = Tensor::stack(&nlls, 0)?;
    println!("==========");
    println!(
        "average negative log likelihood: {:?}",
        nlls.mean(0)?.to_vec0::<f32>()?
    );

    Ok(())
}
