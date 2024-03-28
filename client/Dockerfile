# Use an official Node runtime as a parent image
FROM node:20-alpine

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY ./client /usr/src/app

# Install any needed packages
RUN npm install

# Build the Vite app
RUN npm run build

# Install serve to run the production build
RUN npm install -g serve

# Make port 3000 available to the world outside this container
EXPOSE 3000

# Run the server
CMD ["serve", "-s", "dist", "-l", "3000"]
