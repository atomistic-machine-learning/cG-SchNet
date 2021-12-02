import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import Iterable

import schnetpack as spk
from schnetpack.nn import MLP
from schnetpack.metrics import Metric


### OUTPUT MODULE ###
class AtomwiseWithProcessing(nn.Module):
    r"""
    Atom-wise dense layers that allow to use additional pre- and post-processing layers.

    Args:
        n_in (int, optional): input dimension of representation (default: 128)
        n_out (int, optional): output dimension (default: 1)
        n_layers (int, optional): number of atom-wise dense layers in output network
            (default: 5)
        n_neurons (list of int or int or None, optional): number of neurons in each
            layer of the output network. If a single int is provided, all layers will
            have that number of neurons, if `None`, interpolate linearly between n_in
            and n_out (default: None).
        activation (function, optional): activation function for hidden layers
            (default: spk.nn.activations.shifted_softplus).
        preprocess_layers (nn.Module, optional): a torch.nn.Module or list of Modules
            for preprocessing the representation given by the first part of the network
            (default: None).
        postprocess_layers (nn.Module, optional): a torch.nn.Module or list of Modules
            for postprocessing the output given by the second part of the network
            (default: None).
        in_key (str, optional): keyword to access the representation in the inputs
            dictionary, it is automatically inferred from the preprocessing layers, if
            at least one is given (default: 'representation').
        out_key (str, optional): a string as key to the output dictionary (if set to
            'None', the output will not be wrapped into a dictionary, default: 'y')

    Returns:
        result: dictionary with predictions stored in result[out_key]
    """

    def __init__(self, n_in=128, n_out=1, n_layers=5, n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 preprocess_layers=None, postprocess_layers=None,
                 in_key='representation', out_key='y'):

        super(AtomwiseWithProcessing, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = n_layers
        self.in_key = in_key
        self.out_key = out_key

        if isinstance(preprocess_layers, Iterable):
            self.preprocess_layers = nn.ModuleList(preprocess_layers)
            self.in_key = self.preprocess_layers[-1].out_key
        elif preprocess_layers is not None:
            self.preprocess_layers = preprocess_layers
            self.in_key = self.preprocess_layers.out_key
        else:
            self.preprocess_layers = None

        if isinstance(postprocess_layers, Iterable):
            self.postprocess_layers = nn.ModuleList(postprocess_layers)
        else:
            self.postprocess_layers = postprocess_layers

        if n_neurons is None:
            # linearly interpolate between n_in and n_out
            n_neurons = list(np.linspace(n_in, n_out, n_layers + 1).astype(int)[1:-1])
        self.out_net = MLP(n_in, n_out, n_neurons, n_layers, activation)

        self.derivative = None  # don't compute derivative w.r.t. inputs

    def forward(self, inputs):
        """
        Compute layer output and apply pre-/postprocessing if specified.

        Args:
            inputs (dict of torch.Tensor): batch of input values.
        Returns:
            torch.Tensor: layer output.
        """
        # apply pre-processing layers
        if self.preprocess_layers is not None:
            if isinstance(self.preprocess_layers, Iterable):
                for pre_layer in self.preprocess_layers:
                    inputs = pre_layer(inputs)
            else:
                inputs = self.preprocess_layers(inputs)

        # get (pre-processed) representation
        if isinstance(inputs[self.in_key], tuple):
            repr = inputs[self.in_key][0]
        else:
            repr = inputs[self.in_key]

        # apply output network
        result = self.out_net(repr)

        # apply post-processing layers
        if self.postprocess_layers is not None:
            if isinstance(self.postprocess_layers, Iterable):
                for post_layer in self.postprocess_layers:
                    result = post_layer(inputs, result)
            else:
                result = self.postprocess_layers(inputs, result)

        # use provided key to store result
        if self.out_key is not None:
            result = {self.out_key: result}

        return result


class RepresentationConditioning(nn.Module):
    r"""
    Layer that allows to alter the extracted feature representations in order to
    condition generation. Takes multiple networks that provide conditioning
    information as vectors, stacks these vectors and processes them in a fully
    connected MLP to get a global conditioning vector that is incorporated into
    the extracted feature representation.

    Args:
        layers (nn.Module): a torch.nn.Module or list of Modules that each provide a
            vector representing information for conditioning.
        mode (str, optional): how to incorporate the global conditioning vector in
            the extracted feature representation (can either be 'multiplication',
            'addition', or 'stack', default: 'stack').
        n_global_cond_features (int, optional): number of features in the global
            conditioning vector (i.e. output dimension for the MLP used to aggregate
            the stacked separate conditioning vectors).
        n_layers (int, optional): number of dense layers in the MLP used to get the
            global conditioning vector (default: 5).
        n_neurons (list of int or int or None, optional): number of neurons in each
            layer of the MLP. If a single int is provided, all layers will have that
            number of neurons, if `None`, interpolate linearly between n_in and n_out
            (default: None).
        activation (function, optional): activation function for hidden layers in the
            aggregation MLP (default: spk.nn.activations.shifted_softplus).
        in_key (str, optional): keyword to access the representation in the inputs
            dictionary, it is automatically inferred from the preprocessing layers, if
            at least one is given (default: 'representation').
        out_key (str, optional): a string as key to the output dictionary (if set to
            'None', the output will not be wrapped into a dictionary, default:
            'representation')

    Returns:
        result: dictionary with predictions stored in result[out_key]
    """

    def __init__(self,
                 layers,
                 mode='stack',
                 n_global_cond_features=128,
                 n_layers=5,
                 n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 in_key='representation',
                 out_key='representation'):

        super(RepresentationConditioning, self).__init__()

        if type(layers) not in [list, nn.ModuleList]:
            layers = [layers]
        if type(layers) == list:
            layers = nn.ModuleList(layers)
        self.layers = layers
        self.mode = mode
        self.in_key = in_key
        self.out_key = out_key
        self.n_global_cond_features = n_global_cond_features

        self.derivative = None  # don't compute derivative w.r.t. inputs

        # set number of additional features
        self.n_additional_features = 0
        if self.mode == 'stack':
            self.n_additional_features = self.n_global_cond_features

        # compute number of inputs to the MLP processing stacked conditioning vectors
        n_in = 0
        for layer in self.layers:
            n_in += layer.n_out
        n_out = n_global_cond_features

        # initialize MLP processing stacked conditioning vectors
        if n_neurons is None:
            # linearly interpolate between n_in and n_out
            n_neurons = list(np.linspace(n_in, n_out, n_layers + 1).astype(int)[1:-1])
        self.cond_mlp = MLP(n_in, n_out, n_neurons, n_layers, activation)

    def forward(self, inputs):
        """
        Update representation in the inputs according to conditioning information and
        return empty dictionary since no proper network output is computed in this
        module.

        Args:
            inputs (dict of torch.Tensor): batch of input values.
        Returns:
            dict: An empty dictionary.
        """
        # get (pre-processed) representation
        if isinstance(inputs[self.in_key], tuple):
            repr = inputs[self.in_key][0]
        else:
            repr = inputs[self.in_key]

        # get mask that (potentially) hides conditional information
        _size = [1, len(self.layers)] + [1 for _ in repr.size()[1:]]
        if '_cond_mask' in inputs:
            cond_mask = inputs['_cond_mask']
            cond_mask = cond_mask.reshape([cond_mask.shape[0]] + _size[1:])
        else:
            cond_mask = torch.ones(_size, dtype=repr.dtype, device=repr.device)

        # get conditioning information vectors from layers and include them in
        # representation
        cond_vecs = []
        for i, layer in enumerate(self.layers):
            cond_vecs += [cond_mask[:, i] * layer(inputs)]

        cond_vecs = torch.cat(cond_vecs, dim=-1)
        final_cond_vec = self.cond_mlp(cond_vecs)

        if self.mode == 'addition':
            repr = repr + final_cond_vec
        elif self.mode == 'multiplication':
            repr = repr * final_cond_vec
        elif self.mode == 'stack':
            repr = torch.cat([repr, final_cond_vec.expand(*repr.size()[:-1], -1)], -1)

        inputs.update({self.out_key: repr})

        return {}


### METRICS ###
class KLDivergence(Metric):
    r"""
    Metric for mean KL-Divergence.

    Args:
        target (str, optional): name of target property (default: '_labels')
        model_output (list of int or list of str, optional): indices or keys to unpack
            the desired output from the model in case of multiple outputs, e.g.
            ['x', 'y'] to get output['x']['y'] (default: 'y').
        name (str, optional): name used in logging for this metric. If set to `None`,
            `KLD_[target]` will be used (default: None).
        mask (str, optional): key for a mask in the examined batch which hides
            irrelevant output values. If 'None' is provided, no mask will be applied
            (default: None).
        inverse_mask (bool, optional): whether the mask needs to be inverted prior to
            application (default: False).
    """

    def __init__(self, target='_labels', model_output='y', name=None,
                 mask=None, inverse_mask=False):
        name = 'KLD_' + target if name is None else name
        super(KLDivergence, self).__init__(name)
        self.target = target
        self.model_output = model_output
        self.loss = 0.
        self.n_entries = 0.
        self.mask_str = mask
        self.inverse_mask = inverse_mask

    def reset(self):
        self.loss = 0.
        self.n_entries = 0.

    def add_batch(self, batch, result):
        # extract true labels
        y = batch[self.target]

        # extract predictions
        yp = result
        if self.model_output is not None:
            if isinstance(self.model_output, list):
                for key in self.model_output:
                    yp = yp[key]
            else:
                yp = yp[self.model_output]

        # normalize output
        log_yp = F.log_softmax(yp, -1)

        # apply KL divergence formula entry-wise
        loss = F.kl_div(log_yp, y, reduction='none')

        # sum over last dimension to get KL divergence per distribution
        loss = torch.sum(loss, -1)

        # apply mask to filter padded dimensions
        if self.mask_str is not None:
            atom_mask = batch[self.mask_str]
            if self.inverse_mask:
                atom_mask = 1.-atom_mask
            loss = torch.where(atom_mask > 0, loss, torch.zeros_like(loss))
            n_entries = torch.sum(atom_mask > 0)
        else:
            n_entries = torch.prod(torch.tensor(loss.size()))

        # calculate loss and n_entries
        self.n_entries += n_entries.detach().cpu().data.numpy()
        self.loss += torch.sum(loss).detach().cpu().data.numpy()

    def aggregate(self):
        return self.loss / max(self.n_entries, 1.)


### PRE- AND POST-PROCESSING LAYERS ###
class EmbeddingMultiplication(nn.Module):
    r"""
    Layer that multiplies embeddings of given types with the representation.

    Args:
        embedding (torch.nn.Embedding instance): the embedding layer used to embed atom
            types.
        in_key_types (str, optional): the keyword to obtain types for embedding from
            inputs.
        in_key_representation (str, optional): the keyword to obtain the representation
            from inputs.
        out_key (str, optional): the keyword used to store the calculated product in
            the inputs dictionary.
    """

    def __init__(self, embedding, in_key_types='_next_types',
                 in_key_representation='representation',
                 out_key='preprocessed_representation'):
        super(EmbeddingMultiplication, self).__init__()
        self.embedding = embedding
        self.in_key_types = in_key_types
        self.in_key_representation = in_key_representation
        self.out_key = out_key

    def forward(self, inputs):
        """
        Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values containing the atomic
                numbers for embedding as well as the representation.
        Returns:
            torch.Tensor: layer output.
        """
        # get types to embed from inputs
        types = inputs[self.in_key_types]
        st = types.size()

        # embed types
        if len(st) == 1:
            emb = self.embedding(types.view(st[0], 1))
        elif len(st) == 2:
            emb = self.embedding(types.view(*st[:-1], 1, st[-1]))

        # get representation
        if isinstance(inputs[self.in_key_representation], tuple):
            repr = inputs[self.in_key_representation][0]
        else:
            repr = inputs[self.in_key_representation]
        if len(st) == 2:
            # if multiple types are provided per molecule, expand
            # dimensionality of representation
            repr = repr.view(*repr.size()[:-1], 1, repr.size()[-1])

        # if representation is larger than the embedding, pad embedding with ones
        if repr.size()[-1] != emb.size()[-1]:
            _emb = torch.ones([*emb.size()[:-1], repr.size()[-1]], device=emb.device)
            _emb[..., :emb.size()[-1]] = emb
            emb = _emb

        # multiply embedded types with representation
        features = repr * emb

        # store result in input dictionary
        inputs.update({self.out_key: features})

        return inputs


class NormalizeAndAggregate(nn.Module):
    r"""
    Layer that normalizes and aggregates given input along specifiable axes.

    Args:
        normalize (bool, optional): set True to normalize the input (default: True).
        normalization_axis (int, optional): axis along which normalization is applied
            (default: -1).
        normalization_mode (str, optional): which normalization to apply (currently
            only 'logsoftmax' is supported, default: 'logsoftmax').
        aggregate (bool, optional): set True to aggregate the input (default: True).
        aggregation_axis (int, optional): axis along which aggregation is applied
            (default: -1).
        aggregation_mode (str, optional): which aggregation to apply (currently 'sum'
            and 'mean' are supported, default: 'sum').
        keepdim (bool, optional): set True to keep the number of dimensions after
            aggregation (default: True).
        in_key_mask (str, optional): key to extract a mask from the inputs dictionary,
            which hides values during aggregation (default: None).
        squeeze (bool, optional): whether to squeeze the input before applying
            normalization (default: False).

    Returns:
        torch.Tensor: input after normalization and aggregation along specified axes.
    """

    def __init__(self, normalize=True, normalization_axis=-1,
                 normalization_mode='logsoftmax', aggregate=True,
                 aggregation_axis=-1, aggregation_mode='sum', keepdim=True,
                 mask=None, squeeze=False):

        super(NormalizeAndAggregate, self).__init__()

        if normalize:
            if normalization_mode.lower() == 'logsoftmax':
                self.normalization = nn.LogSoftmax(normalization_axis)
        else:
            self.normalization = None

        if aggregate:
            if aggregation_mode.lower() == 'sum':
                self.aggregation =\
                    spk.nn.base.Aggregate(aggregation_axis, mean=False,
                                          keepdim=keepdim)
            elif aggregation_mode.lower() == 'mean':
                self.aggregation =\
                    spk.nn.base.Aggregate(aggregation_axis, mean=True,
                                          keepdim=keepdim)
        else:
            self.aggregation = None

        self.mask = mask
        self.squeeze = squeeze

    def forward(self, inputs, result):
        """
        Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values containing the mask
            result (torch.Tensor): batch of result values to which normalization and
                aggregation is applied
        Returns:
            torch.Tensor: normalized and aggregated result.
        """

        res = result

        if self.squeeze:
            res = torch.squeeze(res)

        if self.normalization is not None:
            res = self.normalization(res)

        if self.aggregation is not None:
            if self.mask is not None:
                mask = inputs[self.mask]
            else:
                mask = None
            res = self.aggregation(res, mask)

        return res


class AtomCompositionEmbedding(nn.Module):
    r"""
    Layer that embeds all atom types in a molecule and aggregates them into a single
    representation of the composition using a fully connected MLP.

    Args:
        embedding (torch.nn.Embedding instance): an embedding layer used to embed atom
            types separately.
        n_out (int, optional): number of features in the final, global embedding (i.e.
            output dimension for the MLP used to aggregate the separate, stacked atom
            type embeddings).
        n_layers (int, optional): number of dense layers used to get the global
            embedding (default: 5).
        n_neurons (list of int or int or None, optional): number of neurons in each
            layer of the aggregation MLP. If a single int is provided, all layers will
            have that number of neurons, if `None`, interpolate linearly between n_in
            and n_out (default: None).
        activation (function, optional): activation function for hidden layers in the
            aggregation MLP (default: spk.nn.activations.shifted_softplus).
        type_weighting (str, optional): how to weight the individual atom type
            embeddings (choose from 'absolute' to multiply each embedding with the
            absolute number of atoms of that type, 'relative' to multiply with the
            fraction of atoms of that type, and 'existence' to multiply with one if the
            type is present in the composition and zero otherwise, default: 'absolute')
        in_key_composition (str, optional): the keyword to obtain the global
            composition of molecules (i.e. a list of all atom types, default:
            'composition').
        n_types (int, optional): total number of available atom types (default: 5).
    """

    def __init__(self,
                 embedding,
                 n_out=128,
                 n_layers=5,
                 n_neurons=None,
                 activation=spk.nn.activations.shifted_softplus,
                 type_weighting='exact',
                 in_key_composition='composition',
                 n_types=5,
                 skip_h=True):

        super(AtomCompositionEmbedding, self).__init__()

        self.embedding = embedding
        self.in_key_composition = in_key_composition
        self.type_weighting = type_weighting
        self.n_types = n_types
        self.skip_h = skip_h
        if self.skip_h:
            self.n_types -= 1
        self.n_out = n_out

        # compute number of features in stacked embeddings
        n_in = self.n_types * self.embedding.embedding_dim

        if n_neurons is None:
            # linearly interpolate between n_in and n_out
            n_neurons = list(np.linspace(n_in, n_out, n_layers + 1).astype(int)[1:-1])
        self.aggregation_mlp = MLP(n_in, n_out, n_neurons, n_layers, activation)

    def forward(self, inputs):
        """
        Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values containing the atomic
                numbers for embedding as well as the representation.
        Returns:
            torch.Tensor: batch of vectors representing the global composition of
                each molecule.
        """
        # get composition to embed from inputs
        compositions = inputs[self.in_key_composition][..., None]
        if self.skip_h:
            embeded_types = self.embedding(inputs['_all_types'][0, 1:-1])[None, ...]
        else:
            embeded_types = self.embedding(inputs['_all_types'][0, :-1])[None, ...]

        # get global representation
        if self.type_weighting == 'relative':
            compositions = compositions/torch.sum(compositions, dim=-2, keepdim=True)
        elif self.type_weighting == 'existence':
            compositions = (compositions > 0).float()

        # multiply embedding with (weighted) composition
        embedding = embeded_types * compositions

        # aggregate embeddings to global representation
        sizes = embedding.size()
        embedding = embedding.view([*sizes[:-2], 1, sizes[-2]*sizes[-1]])  # stack
        embedding = self.aggregation_mlp(embedding)  # aggregate

        return embedding


class FingerprintEmbedding(nn.Module):
    r"""
    Layers that map the fingerprint of a molecule to a feature vector used for
    conditioning.

    Args:
        n_in (int): number of inputs (bits in the fingerprint).
        n_out (str): number of features in the embedding.
        n_layers (int, optional): number of dense layers used to embed the fingerprint
            (default: 5).
        n_neurons (list of int or int or None, optional): number of neurons in each
            layer of the output network. If a single int is provided, all layers will
            have that number of neurons, if `None`, interpolate linearly between n_in
            and n_out (default: None).
        in_key_fingerprint (str, optional): the keyword to obtain the fingerprint
            (default: 'fingerprint').
        activation (function, optional): activation function for hidden layers
            (default: spk.nn.activations.shifted_softplus).
    """

    def __init__(self, n_in, n_out, n_layers=5, n_neurons=None,
                 in_key_fingerprint='fingerprint',
                 activation=spk.nn.activations.shifted_softplus):

        super(FingerprintEmbedding, self).__init__()

        self.in_key_fingerprint = in_key_fingerprint
        self.n_in = n_in
        self.n_out = n_out

        if n_neurons is None:
            # linearly interpolate between n_in and n_out
            n_neurons = list(np.linspace(n_in, n_out, n_layers + 1).astype(int)[1:-1])
        self.out_net = MLP(n_in, n_out, n_neurons, n_layers, activation)

    def forward(self, inputs):
        """
        Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values containing the
                fingerprints.
        Returns:
            torch.Tensor: batch of vectors representing the fingerprint of each
                molecule.
        """
        fingerprints = inputs[self.in_key_fingerprint]

        return self.out_net(fingerprints)[:, None, :]


class PropertyEmbedding(nn.Module):
    r"""
    Layers that map the property (e.g. HOMO-LUMO gap, electronic spatial extent etc.)
    of a molecule to a feature vector used for conditioning. Properties are first
    expanded using Gaussian basis functions before being processed by a fully
    connected MLP.

    Args:
        n_in (int): number of inputs (Gaussians used for expansion of the property).
        n_out (int): number of features in the embedding.
        in_key_property (str): the keyword to obtain the property.
        start (float): center of first Gaussian function, :math:`\mu_0` for expansion.
        stop (float): center of last Gaussian function, :math:`\mu_{N_g}` for expansion
            (the remaining centers will be placed linearly spaced between start and
            stop).
        n_layers (int, optional): number of dense layers used to embed the property
            (default: 5).
        n_neurons (list of int or int or None, optional): number of neurons in each
            layer of the output network. If a single int is provided, all layers will
            have that number of neurons, if `None`, interpolate linearly between n_in
            and n_out (default: None).
        activation (function, optional): activation function for hidden layers
            (default: spk.nn.activations.shifted_softplus).
        trainable_gaussians (bool, optional): if True, widths and offset of Gaussian
            functions for expansion are adjusted during training process (default:
            False).
        widths (float, optional): width value of Gaussian functions for expansion
            (provide None to set the width to the distance between two centers
            :math:`\mu`, default: None).
    """

    def __init__(self, n_in, n_out, in_key_property, start, stop, n_layers=5,
                 n_neurons=None, activation=spk.nn.activations.shifted_softplus,
                 trainable_gaussians=False, width=None, no_expansion=False):

        super(PropertyEmbedding, self).__init__()

        self.in_key_property = in_key_property
        self.n_in = n_in
        self.n_out = n_out
        if not no_expansion:
            self.expansion_net = GaussianExpansion(start, stop, self.n_in,
                                                   trainable_gaussians, width)
        else:
            self.expansion_net = None

        if n_neurons is None:
            # linearly interpolate between n_in and n_out
            n_neurons = list(np.linspace(n_in, n_out, n_layers + 1).astype(int)[1:-1])
        self.out_net = MLP(n_in, n_out, n_neurons, n_layers, activation)

    def forward(self, inputs):
        """
        Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values containing the
                fingerprints.
        Returns:
            torch.Tensor: batch of vectors representing the fingerprint of each
                molecule.
        """
        property = inputs[self.in_key_property]
        if self.expansion_net is None:
            expanded = property
        else:
            expanded = self.expansion_net(property)

        return self.out_net(expanded)[:, None, :]


### MISC
class GaussianExpansion(nn.Module):
    r"""Expansion layer using a set of Gaussian functions.

    Args:
        start (float): center of first Gaussian function, :math:`\mu_0`.
        stop (float): center of last Gaussian function, :math:`\mu_{N_g}`.
        n_gaussians (int, optional): total number of Gaussian functions, :math:`N_g`
            (default: 50).
        trainable (bool, optional): if True, widths and offset of Gaussian functions
            are adjusted during training process (default: False).
        widths (float, optional): width value of Gaussian functions (provide None to
            set the width to the distance between two centers :math:`\mu`, default:
            None).

    """

    def __init__(self, start, stop, n_gaussians=50, trainable=False,
                 width=None):
        super(GaussianExpansion, self).__init__()
        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, stop, n_gaussians)
        if width is None:
            widths = torch.FloatTensor((offset[1] - offset[0]) *
                                       torch.ones_like(offset))
        else:
            widths = torch.FloatTensor(width * torch.ones_like(offset))
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, property):
        """Compute expanded gaussian property values.

        Args:
            property (torch.Tensor): property values of (N_b x 1) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_g) shape.

        """
        # compute width of Gaussian functions (using an overlap of 1 STDDEV)
        coeff = -0.5 / torch.pow(self.widths, 2)[None, :]
        # Use advanced indexing to compute the individual components
        diff = property - self.offsets[None, :]
        # compute expanded property values
        return torch.exp(coeff * torch.pow(diff, 2))
