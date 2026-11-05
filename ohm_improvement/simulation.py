class PCMSimulator(nn.Module):
    def __init__(self, model, templates, scales, noise, compensation):
        super().__init__()
        # model must be separate per each PCMSimulator instance
        self.model = model
        self.model.requires_grad_(False)
        self.scales = tuple(scales)
        self.noise = noise
        self.compensation = compensation

        # layers hard-coded for now
        for i, layer in enumerate((self.model.l1, self.model.l2)):
            tile = next(layer.analog_tiles())
            if "shared_weights" in tile._parameters:
                del tile._parameters["shared_weights"]

            for name in ("program", "drift", "read"):
                for side in ("plus", "minus"):
                    self.register_buffer(
                        f"{name}_{side}_{i}",
                        torch.randn_like(templates[i]),
                    )

    def analog_forward(self, tile, inputs):
        # This bypasses learned output scaling.
        return AnalogFunction.apply(
            tile.get_analog_ctx(), tile, inputs,
            tile.shared_weights, True,
        )

    def simulate_layer(self, i, layer, inputs, g_plus, g_minus, t):
        # Assume one tile per layer
        tile = next(layer.analog_tiles())

        def sample(name, side):
            return getattr(self, f"{name}_{side}_{i}")

        def install(gp, gm):
            weights = differential_to_logical(
                gp, gm, self.scales[i]
            )

            if weights.is_cuda:
                weights = weights.transpose(0, 1)

            tile.shared_weights = weights.contiguous()

        gp = self.noise.apply_programming_noise_to_conductance(
            g_plus, sample("program", "plus")
        )
        gm = self.noise.apply_programming_noise_to_conductance(
            g_minus, sample("program", "minus")
        )

        nu_plus = self.noise.generate_drift_coefficients(
            g_plus, sample("drift", "plus")
        )
        nu_minus = self.noise.generate_drift_coefficients(
            g_minus, sample("drift", "minus")
        )

        install(gp, gm)
        reference = self.compensation.init_baseline(tile)

        gp = self.noise.apply_drift_noise_to_conductance(
            gp, nu_plus, t, sample("read", "plus")
        )
        gm = self.noise.apply_drift_noise_to_conductance(
            gm, nu_minus, t, sample("read", "minus")
        )
        install(gp, gm)

        probe = self.compensation.get_readout_tensor(
            tile.in_size
        ).to(tile.shared_weights)
        probe_output = self.analog_forward(tile, probe)
        output = self.analog_forward(tile, inputs)

        output = self.compensation.apply(
            probe_output, reference, output
        )

        if tile.digital_bias:
            output = output + tile.bias

        return output

    def forward(self, x, g_plus, g_minus, t_inference):
        # Hard-coded for now
        x = x.reshape(-1, 784)
        # layers hard-coded for now
        for i, layer in enumerate((self.model.l1, self.model.l2)):
            x = self.simulate_layer(
                i, layer, x, g_plus[i], g_minus[i], t_inference
            )
            x = F.relu(x)

        return F.log_softmax(x, dim=1)


class PCMEnsemble(nn.Module):
    def __init__(self, pairs, simulators, t_inferences):
        super().__init__()
        self.g_plus = nn.ParameterList([
            nn.Parameter(gp.detach().clone()) for gp, gm in pairs
        ])
        self.g_minus = nn.ParameterList([
            nn.Parameter(gm.detach().clone()) for gp, gm in pairs
        ])
        self.simulators = nn.ModuleList(simulators)
        self.t_inferences = tuple(t_inferences)

    def forward(self, x):
        return torch.stack([
            torch.stack([
                simulator(x, self.g_plus, self.g_minus, t)
                for t in self.t_inferences
            ])
            for simulator in self.simulators
        ])
