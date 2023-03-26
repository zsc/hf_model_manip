import copy
import numpy
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM

def random_orthogonal_matrix(n):
    '''generate a random orthogonal matrix'''
    a = numpy.random.normal(0.0, 1.0, (n, n))
    q, r = numpy.linalg.qr(a)
    return q

def compress(name, model, ratio, last_layer=29):
    '''multiply random orthogonal matrices to the paring weights of the MLP of the transformer to compress the model'''

    # only keep two significant digits for ratio
    compressed_name = f'{name}-compressed-{int(ratio * 100)}'

    # load existing compressed model if exists
    try:
        model2 = BloomForCausalLM.from_pretrained(compressed_name)
        return model2
    except:
        pass

    configuration = copy.deepcopy(model.config)
    configuration.up_scale_factor = 4 * ratio
    model2 = BloomForCausalLM(configuration)

    state_dict = model.state_dict().copy()
    for name, param in model.named_parameters():
        #print(name, param.shape)
        #continue

        # skip the first and last transformer block
        #if 'transformer.h.0' in name or f'transformer.h.{last_layer}' in name:
        #    continue
        if 'mlp' in name and 'dense_4h_to_h.weight' in name:
            print(name, param.shape)
            m = random_orthogonal_matrix(param.shape[1])
            # extract first ratio * 4h weights
            m = m[:int(ratio * param.shape[1])]
            # turn m into a float16 Tensor
            m = torch.from_numpy(m).float()

            state_dict[name] = param.data @ m.T
            print('=>', name, state_dict[name].shape)


            # get the pairing h_to_4h weights
            name2 = name.replace('dense_4h_to_h.weight', 'dense_h_to_4h.weight')
            param2 = model.state_dict()[name2]
            print(name2, param2.shape)
            state_dict[name2] = m @ param2.data
            print('=>', name2, state_dict[name2].shape)

            # also truncate the accompanying bias
            name2 = name.replace('dense_4h_to_h.weight', 'dense_h_to_4h.bias')
            param2 = model.state_dict()[name2]
            print(name2, param2.shape)
            state_dict[name2] = param2.data[:int(ratio * param2.shape[0])]
            print('=>', name2, state_dict[name2].shape)

            #break

    #state_dict.pop('lm_head.weight')
    #model2.load_state_dict({k.replace('transformer.', ''):v for k, v in state_dict.items()})
    model2.load_state_dict(state_dict)
    model2.save_pretrained(compressed_name)
    return model2
    #from IPython import embed; embed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="bigscience/bloom-560m")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=30)
    args = parser.parse_args()

    '''
    m = random_orthogonal_matrix(10)
    # check if m is nearly orthogonal using allclose
    print(numpy.allclose(m @ m.T, numpy.eye(10)))
    '''

    # load a Hugging face mode using `from_pretrained`
    model_id = args.model_id
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text = """OpenAI is a company"""
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(tokenizer.batch_decode(model.generate(input_ids)))
    # multinomial sampling
    for x in tokenizer.batch_decode(model.generate(input_ids, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=3)):
        print(x)
        print()

    # compress the model
    model2 = compress(model_id.replace('/', '_'), model, args.ratio, last_layer=args.num_layers-1)
    print(tokenizer.batch_decode(model2.generate(input_ids)))
    for x in tokenizer.batch_decode(model2.generate(input_ids, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=3)):
        print(x)
        print()

