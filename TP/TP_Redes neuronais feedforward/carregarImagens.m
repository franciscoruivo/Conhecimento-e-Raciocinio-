function [imagens, targets] = carregarImagens(pasta)
% CARREGARIMAGENS - Carrega as imagens da pasta especificada
% 
% Parâmetros:
%   pasta - Nome da pasta (start, train ou test)
%
% Retorna:
%   imagens - Matriz de imagens (cada coluna é uma imagem redimensionada)
%   targets - Matriz de targets (codificação one-hot)

    % Definir classes
    classes = {'circle', 'kite', 'parallelogram', 'square', 'trapezoid', 'triangle'};
    num_classes = length(classes);
    
    % Verificar se pasta é válida
    % pastas_validas = {'start', 'train', 'test', 'imagens_desenhadas'};
    % if ~ismember(pasta, pastas_validas)
    %     error('Pasta inválida. Use "start", "train","test" ou imagens desenhadas.');
    % end
    
    % Tamanho para redimensionar imagens (mesmo usado no treino)
    tam_img = [28 28];
    
    % Inicializar vetores para armazenar imagens e targets
    imagens = [];
    targets = [];
    
    % Carregar imagens de cada classe
    for c = 1:num_classes
        % Caminho para a pasta da classe
        classe_path = fullfile(pasta, classes{c});
        
        % Listar arquivos na pasta
        files = dir(fullfile(classe_path, '*.png'));
        
        for f = 1:length(files)
            % Caminho completo do arquivo
            file_path = fullfile(classe_path, files(f).name);
            
            % Carregar imagem
            img = imread(file_path);
            
            % Converter para escala de cinza se for colorida
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            
            % Binarizar imagem
            img = imbinarize(img);
            
            % Redimensionar imagem
            img_resized = imresize(img, tam_img);
            
            % Converter para vetor coluna
            img_vec = img_resized(:);
            
            % Adicionar à matriz de imagens
            imagens = [imagens, double(img_vec)];
            
            % Criar target (one-hot encoding)
            target = zeros(num_classes, 1);
            target(c) = 1;
            
            % Adicionar à matriz de targets
            targets = [targets, target];
        end
    end
end
