import os
import sys
import time
import argparse
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from modulesLstm import generator_module, discriminator_module
from preprocessing import preprocessingFirstColumn
from Data.Dataset import make_train_data
import torch.nn as nn
import modules
import pandas as pd
import pickle
from inversetranformation import revert_encoding
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--output_dir', type=str, default='./pretrainedRNN_models/', help='output directory')
    parser.add_argument('--sample_data', type=str, default='./example_data/sample_data.txt', help='Data required to be mimced')
    parser.add_argument('--device', type=str, default='cpu',help='No CUDA cpu else cuda',required=False)
    parser.add_argument('--cell_type', type=str, default='lstm', help='The type of cells :  lstm or gru')
    # dataset parameters
    parser.add_argument('--batch_size', type=int, default=25, help='batch size used during training')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the LSTM')
    parser.add_argument('--hidden_size', type=int, default=128, help='number of hidden diemensions')
    parser.add_argument('--number_features', type=int, default=47, help='number of features in dataset')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='epochs')
    args = parser.parse_args()


    dataset = pd.read_csv(args.sample_data)
    samples,number_Features = obtain_shape = dataset.shape
    encodedDataframe = preprocessingFirstColumn(args)
    train_loader = make_train_data(args,encodedDataframe)

    if args.cell_type == 'gru':
        from modulesGru import generator_module, discriminator_module
        generator_mod = generator_module(args)#.to(device=args.device)
        discriminator_mod = discriminator_module(args)#.to(device=args.device)
        optimizer_discriminator = torch.optim.Adam(discriminator_mod.parameters(), lr=args.learning_rate)#.to(device=args.device)
        optimizer_generator = torch.optim.Adam(generator_mod.parameters(), lr=args.learning_rate)#.to(device=args.device)

    else:
        generator_mod = generator_module(args)  # .to(device=args.device)
        discriminator_mod = discriminator_module(args)  # .to(device=args.device)
        optimizer_discriminator = torch.optim.Adam(discriminator_mod.parameters(),lr=args.learning_rate)  # .to(device=args.device)
        optimizer_generator = torch.optim.Adam(generator_mod.parameters(),lr=args.learning_rate)  # .to(device=args.device)


    loss_function = nn.BCELoss()
    dec_loss = list()
    gen_loss = list()
    for epoch in range(args.epochs):

        for n, (real_samples, _) in enumerate(train_loader):
            # Data for training the discriminator
            real_samples_labels = torch.ones((args.batch_size, 1), device=torch.device(args.device))
            latent_space_samples = torch.randn((args.batch_size, args.number_features),
                                               device=torch.device(args.device))

            generated_samples = generator_mod(latent_space_samples)
            generated_samples_labels = torch.zeros((args.batch_size, 1), device=torch.device(args.device))
            all_samples = torch.cat((real_samples.to(device=torch.device(args.device)),
                                     generated_samples.to(device=torch.device(args.device))))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels)).to(
                device=torch.device(args.device))

            # Training the discriminator
            discriminator_mod.zero_grad()
            output_discriminator = discriminator_mod(all_samples.float())
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_samples = torch.randn((args.batch_size, args.number_features),
                                               device=torch.device(args.device))
            # Training the generator
            generator_mod.zero_grad()
            generated_samples = generator_mod(latent_space_samples.float())
            output_discriminator_generated = discriminator_mod(generated_samples.float())
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()
            # Show loss
            if epoch % 1 == 0 and n == args.batch_size - 1:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")
                dec_loss.append(loss_discriminator)
                gen_loss.append(loss_generator)

    # Save the modules to specified directory
    torch.save(generator_mod, args.output_dir + "generator.pth")
    torch.save(discriminator_mod, args.output_dir + "discriminator.pth")

    #Load the preprocessed model templates for inverse transformations
    local_labelencoded = pickle.load(open(args.output_dir+'label_encoder.pickle', 'rb'))
    local_onehot_encoder = pickle.load(open(args.output_dir+'onehot_encoder.pickle', 'rb'))
    local_scaler = pickle.load(open(args.output_dir + 'standard_scaler.pickle', 'rb'))
    local_pca = pickle.load(open(args.output_dir + 'pca.pickle', 'rb'))

    #Genererae synthetic data from the samples
    latent_space_samples = torch.randn((samples, args.number_features), device=torch.device(args.device))
    generated_samples = generator_mod(latent_space_samples)

    # Convert variables back to cpu if cuda is enabled
    generated_samples2 = generated_samples.clone().detach().to(device=torch.device('cpu'))

    gen_rescaled_sample = local_scaler.inverse_transform(generated_samples2)
    gen_rescaled_sample = local_pca.inverse_transform(gen_rescaled_sample)
    regeneratedCellID = revert_encoding(gen_rescaled_sample[:, :21],local_labelencoded)

    #Reconstruct the dataframe from the samples
    reconDataframe = pd.DataFrame(data=np.array(gen_rescaled_sample[:, 21:]),columns=['Velocity', 'Load', 'EdgeUsersDist', 'Hysteresis', 'TTT',
                               'CIO(3)', 'CIO(4)', 'CIO(5)', 'CIO(6)', 'CIO(7)', 'CIO(8)', 'CIO(9)',
                               'CIO(10)', 'CIO(11)', 'CIO(12)', 'CIO(13)', 'CIO(14)', 'CIO(15)',
                               'CIO(16)', 'CIO(17)', 'CIO(18)', 'CIO(19)', 'CIO(20)', 'CIO(21)',
                               'CIO(22)', 'CIO(23)'])
    reconDataframe['Cell_ID'] = regeneratedCellID

    finalSyntheticData = reconDataframe.reindex(columns=['Cell_ID', 'Velocity', 'Load', 'EdgeUsersDist', 'Hysteresis', 'TTT',
                              'CIO(3)', 'CIO(4)', 'CIO(5)', 'CIO(6)', 'CIO(7)', 'CIO(8)', 'CIO(9)',
                              'CIO(10)', 'CIO(11)', 'CIO(12)', 'CIO(13)', 'CIO(14)', 'CIO(15)',
                              'CIO(16)', 'CIO(17)', 'CIO(18)', 'CIO(19)', 'CIO(20)', 'CIO(21)',
                              'CIO(22)', 'CIO(23)'])

    #Save the output to the directory
    finalSyntheticData.to_csv(args.output_dir+"finalSyntheticData.csv")

    # Save plots of the training
    plt.plot(np.arange(0, args.epochs, 1), np.array(dec_loss), label='Discriminator Loss')
    plt.plot(np.arange(0, args.epochs, 1), np.array(gen_loss), label='Generator Loss')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Convergence for Recurrent Neural GANs')
    plt.legend(loc='upper right')
    plt.savefig(args.output_dir + "Performance_model.png")