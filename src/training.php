<?php

require_once($_SERVER['DOCUMENT_ROOT'] . 'vendor/autoload.php');

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Regressors\DummyRegressor;
use Rubix\ML\CrossValidation\Metrics\SMAPE;
use Rubix\ML\Other\Strategies\Constant;
use Rubix\ML\Persisters\Filesystem;

$trainingData = new CSV(__DIR__.'\datasets\training.csv', true);
$trainingDataset = Labeled::fromIterator($trainingData) -> transformLabels('floatval');

$estimator = new GradientBoost(new RegressionTree(3), 0.1, 0.8, 1000, 1e-4, 10, 0.1, new SMAPE(), new DummyRegressor(new Constant(0.0)));

echo "Modelo em treinamento..." . PHP_EOL;
$estimator -> train($trainingDataset);
echo "Modelo treinado" . PHP_EOL;

echo "Salvando modelo..." . PHP_EOL;
$persister = new Filesystem(__DIR__.'\output\estimator.model');
$persister->save($estimator);
echo "Modelo salvo" . PHP_EOL;

