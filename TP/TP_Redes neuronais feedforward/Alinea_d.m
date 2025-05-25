
% Limpar workspace e fechar todas as figuras
clear all;
close all;
clc;

% Configuração inicial
classes = {'circulo', 'kite', 'paralelograma', 'quadrado', 'trapecoide', 'triangulo'};
pasta_imagens = 'imagens_desenhadas';

arquivos_rede = dir('resultados_alinea_c_Melhores_Redes/rede_*.mat');
redes = cell(1, length(arquivos_rede));
for i = 1:length(arquivos_rede)
    tempStruct = load(fullfile('resultados_alinea_c_Melhores_Redes/', arquivos_rede(i).name));
    vars = fieldnames(tempStruct); 
    redes{i} = tempStruct.(vars{1});
end

% Carregar e processar imagens manuais (assumindo subpastas por classe)
resultados = struct();

% Processar cada categoria
for classIdx = 1:length(classes)
    categoria = classes{classIdx};
    pasta_categoria = fullfile(pasta_imagens, categoria);
    Ficheiros_imagens = dir(fullfile(pasta_categoria, '*.png'));
    
    for imgIdx = 1:length(Ficheiros_imagens)
        % Carregar imagem
        nome_arquivo = fullfile(pasta_categoria, Ficheiros_imagens(imgIdx).name);
        img = imread(nome_arquivo);
        
        % Processar imagem
        if size(img, 3) == 3
            img_gray = rgb2gray(img);
        else
            img_gray = img;
        end
        img_bin = imbinarize(img_gray);
        img_resized = imresize(img_bin, [28 28]); % Ajustar conforme necessário
        
        % Preparar entrada para a rede (ajustar conforme a rede espera)
        if isa(redes{1}, 'SeriesNetwork') || isa(redes{1}, 'DAGNetwork')
            X = im2single(img_resized);
            if ~ismember('ImageInputLayer', {redes{1}.Layers(1).Type})
                X = reshape(X, [28 28 1]);
            end
        else
            X = double(img_resized(:))'; % Vetor linha
        end
        
        % Classificar com cada rede
        for netIdx = 1:length(redes)
            net = redes{netIdx};
            nome_rede = arquivos_rede(netIdx).name(1:end-4);
            
            if isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork')
                [Y_pred, scores] = classify(net, X);
                classe_pred = find(strcmp(classes, char(Y_pred)));
                probabilidades = scores';
            else
                Y_pred = net(X');
                probabilidades = softmax(Y_pred);
                [~, classe_pred] = max(probabilidades);
            end
            
            % Armazenar resultado
            chave = sprintf('%s_%s', nome_rede, Ficheiros_imagens(imgIdx).name(1:end-4));
            resultados.(chave) = struct(...
                'classe_verdadeira', categoria, ...
                'classe_predita', classes{classe_pred}, ...
                'probabilidades', probabilidades ...
            );
        end
    end
end

% Exibir resultados
chaves = fieldnames(resultados);
for i = 1:length(chaves)
    chave = chaves{i};
    fprintf('\nImagem: %s\n', chave);
    res = resultados.(chave);
    fprintf('Classe verdadeira: %s\n', res.classe_verdadeira);
    fprintf('Classe predita: %s\n', res.classe_predita);
    fprintf('Probabilidades:\n');
    for k = 1:length(classes)
        fprintf('  %s: %.2f%%\n', classes{k}, res.probabilidades(k)*100);
    end
end

% Salvar resultados (opcional)
%save('resultados_tarefa_d.mat', 'resultados');