# model_builder.py
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out)  # Dropout 적용
        out = self.fc(out[:, -1, :])
        return out

def build_gru_model(input_size, hidden_size, num_layers, output_size):
    return GRUModel(input_size, hidden_size, num_layers, output_size)

hand_flaping_input_size = 12 * 3
hand_flaping_hidden_size = 32
hand_flaping_num_layers = 1
hand_flaping_output_size = 1

HM_input_size = 42 * 3
HM_hidden_size = 32
HM_num_layers = 1
HM_output_size = 8

PS_input_size = 3 * 3
PS_hidden_size = 32
PS_num_layers = 1
PS_output_size = 3

HandFlapping_GRUModel = build_gru_model(hand_flaping_input_size, hand_flaping_hidden_size, hand_flaping_num_layers, hand_flaping_output_size)
HM_GRUModel = build_gru_model(HM_input_size, HM_hidden_size, HM_num_layers, HM_output_size)
PS_GRUModel = build_gru_model(PS_input_size, PS_hidden_size, PS_num_layers, PS_output_size)
