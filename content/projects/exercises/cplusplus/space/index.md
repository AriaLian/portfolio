+++
title = "Space Shooter"
description = ""
summary = ""
featuredImage = "featured.png"
tags = ["Game Engine"]
draft = false
categories = [""]
collections = ["C++"]
+++

## Overview
This is a parallax Space Shooter game, developed in C++ using the SDL (Simple DirectMedia Layer) library. 
In this top-down space ship shooter game, player can control a spaceship, engage in battles with enemy ships and asteroids, and strive to survive

{{< github repo="aria-0829/Project_01_SpaceShooter" showThumbnail=false >}}

{{< button href="https://github.com/aria-0829/Project_01_SpaceShooter" target="_self" >}}
{{< icon "github" >}} View on Github
{{< /button >}}

## Introduction

### The Player
- The player will use the keyboard controls (WSAD) for spaceship movement and shoot projectiles using the spacebar. 
- The player is capable of destroying enemy ships and asteroids by shooting them with projectiles or by colliding with them (suicide). 
- Player loses lives upon collision with enemies or getting hit by enemy projectiles.
### Enemy Ships
- Sequential spawning of alien enemy ships.
- Different types of enemy ships with distinct actions, include those capable of shooting projectiles. 
- The enemy ships do not collide with asteroids.
- Factory for spawning multiple enemy ships during the game.
### Asteroids
- Different sizes of asteroids floating or flying across the screen, which become dangerous when heading towards you.
- The asteroids are considered to be obstacles that must be avoided or destroyed.
### Game Background
- Stars, clouds and other objects move across the screen, providing a dynamic background.
### Data-Driven Design
- The game loads data from JSON files. 
- Each game element (enemy ship, asteroid, player, etc.) has an associated data file. 
- Data-driven design enables easy modification of textures, lives, health, damage, and other attributes without having to recompile the game.

![spaceshooter](https://github.com/aria-0829/Project_01_SpaceShooter/assets/45384635/08f330c4-582e-4ebd-a1e1-79cfb292abcc)
