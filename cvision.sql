-- phpMyAdmin SQL Dump
-- version 4.8.4
-- https://www.phpmyadmin.net/
--
-- 主機: 127.0.0.1:3306
-- 產生時間： 
-- 伺服器版本: 5.7.24
-- PHP 版本： 7.2.14

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- 資料庫： `cvision`
--

-- --------------------------------------------------------

--
-- 資料表結構 `label`
--

DROP TABLE IF EXISTS `label`;
CREATE TABLE IF NOT EXISTS `label` (
  `id` varchar(50) NOT NULL,
  `name` varchar(100) DEFAULT NULL,
  `model_id` varchar(50) DEFAULT NULL,
  `share` int(1) NOT NULL DEFAULT '0',
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`,`model_id`),
  KEY `label_ibfk_1` (`model_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- 資料表結構 `model`
--

DROP TABLE IF EXISTS `model`;
CREATE TABLE IF NOT EXISTS `model` (
  `id` varchar(50) NOT NULL,
  `key` varchar(50) NOT NULL,
  `name` varchar(100) NOT NULL,
  `size` float DEFAULT NULL,
  `acc` float DEFAULT NULL,
  `loss` float DEFAULT NULL,
  `train_status` tinyint(4) DEFAULT '-1',
  `class_label` text,
  `share` tinyint(1) NOT NULL DEFAULT '0',
  PRIMARY KEY (`id`),
  KEY `key` (`key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- 資料表結構 `predict_list`
--

DROP TABLE IF EXISTS `predict_list`;
CREATE TABLE IF NOT EXISTS `predict_list` (
  `model_id` varchar(50) NOT NULL,
  `result` varchar(100) NOT NULL,
  PRIMARY KEY (`model_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- 資料表結構 `predict_queue`
--

DROP TABLE IF EXISTS `predict_queue`;
CREATE TABLE IF NOT EXISTS `predict_queue` (
  `model_id` varchar(50) NOT NULL,
  `image_id` varchar(50) NOT NULL,
  `ms` varchar(20) NOT NULL
) ENGINE=MyISAM DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- 資料表結構 `progress_list`
--

DROP TABLE IF EXISTS `progress_list`;
CREATE TABLE IF NOT EXISTS `progress_list` (
  `model_id` varchar(50) NOT NULL,
  `epoch` varchar(50) DEFAULT NULL,
  `acc` float DEFAULT NULL,
  `loss` float DEFAULT NULL,
  `pid` int(11) DEFAULT NULL,
  PRIMARY KEY (`model_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- 資料表結構 `train_queue`
--

DROP TABLE IF EXISTS `train_queue`;
CREATE TABLE IF NOT EXISTS `train_queue` (
  `model_id` varchar(50) NOT NULL,
  `ms` varchar(20) NOT NULL,
  PRIMARY KEY (`model_id`)
) ENGINE=MyISAM DEFAULT CHARSET=utf8;

-- --------------------------------------------------------

--
-- 資料表結構 `user`
--

DROP TABLE IF EXISTS `user`;
CREATE TABLE IF NOT EXISTS `user` (
  `key` varchar(50) NOT NULL,
  `email` varchar(50) DEFAULT NULL,
  `name` varchar(64) DEFAULT NULL,
  PRIMARY KEY (`key`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- 資料表的匯出資料 `user`
--

INSERT INTO `user` (`key`, `email`, `name`) VALUES
('1569239820171-4306E9C0-9D01-46BC-84D2-25FD2319772A', 'cvision10904@gmail.com', '109-04 Cjcu-im-prj'),
('1571660970515-B4FBCBAF-DA12-48EA-9858-51A0A55C0465', 'h24563026@mailst.cjcu.edu.tw', 'yiz Zhang'),
('1572174904157-4999DE23-C92D-42A0-A088-41E17DEB5C6C', 'blink105914@gmail.com', '午瑶'),
('1572175003905-8F73B2F7-DB21-413B-8280-52A184F70AE8', '107b19980@mailst.cjcu.edu.tw', '游思敏'),
('1572196828488-732A9438-243A-4B90-A4F7-F75F5BEDD9E7', 'z58774556@gmail.com', '張逸宗');

--
-- 已匯出資料表的限制(Constraint)
--

--
-- 資料表的 Constraints `label`
--
ALTER TABLE `label`
  ADD CONSTRAINT `label_ibfk_1` FOREIGN KEY (`model_id`) REFERENCES `model` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;

--
-- 資料表的 Constraints `model`
--
ALTER TABLE `model`
  ADD CONSTRAINT `model_ibfk_1` FOREIGN KEY (`key`) REFERENCES `user` (`key`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
